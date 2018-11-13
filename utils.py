from itertools import combinations
from torch.autograd import Variable,Function
import numpy as np
import torch
import matplotlib.pyplot as plt

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


def denormalize(tens):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]

    img_1 = tens.clone()
    for t, m, s in zip(img_1, mean, std):
        t.mul_(s).add_(m)
    img_1 = img_1.numpy().transpose(1,2,0)
    return img_1


def display_triplet_distance(model,train_loader,name):
    f, axarr = plt.subplots(3,figsize=(10,10))
    f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_p, data_n,c1,c2) in enumerate(train_loader):

        try:
            data_a_c, data_p_c,data_n_c = data_a.cuda(), data_p.cuda(), data_n.cuda()
            data_a_v, data_p_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_p_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            out_a, out_p, out_n = model(data_a_v), model(data_p_v), model(data_n_v)
        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        print("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0][0]))
        print("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0][0]))


        axarr[0].imshow(denormalize(data_a[0]))
        axarr[1].imshow(denormalize(data_p[0]))
        axarr[2].imshow(denormalize(data_n[0]))
        axarr[0].set_title("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0][0]))
        axarr[2].set_title("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0][0]))

        break
    f.savefig("{}.png".format(name))
    #plt.show()


def display_triplet_distance_test(model,test_loader,name):
    f, axarr = plt.subplots(5,2,figsize=(10,10))
    f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_n,label) in enumerate(test_loader):

        if np.all(label.cpu().numpy()):
            continue

        try:
            data_a_c, data_n_c = data_a.cuda(), data_n.cuda()
            data_a_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            out_a, out_n = model(data_a_v), model(data_n_v)

        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        for i in range(5):
            rand_index = np.random.randint(0, label.size(0)-1)
            if i%2 == 0:
                for j in range(label.size(0)):
                    # Choose label == 0
                    rand_index = np.random.randint(0, label.size(0)-1)
                    if label[rand_index] == 0:
                        break

            distance = l2_dist.forward(out_a,out_n).data[rand_index][0]
            print("Distance: {}".format(distance))
            #distance_pca = l2_dist.forward(PCA(128).fit_transform(out_a.data[i].cpu().numpy()),PCA(128).fit_transform(out_n.data[i].cpu().numpy())).data[0]
            #print("Distance(PCA): {}".format(distance_pca))

            axarr[i][0].imshow(denormalize(data_a[rand_index]))
            axarr[i][1].imshow(denormalize(data_n[rand_index]))
            plt.figtext(0.5, i/5.0+0.1,"Distance : {}, Label: {}\n".format(distance,label[rand_index]), ha='center', va='center')


        break
    plt.subplots_adjust(hspace=0.5)

    f.savefig("{}.png".format(name))
    #plt.show()
