from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import TripletMNIST
from logger import Logger
import torch
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from TripletFaceDataset import TripletFaceDataset
from LFWDataset import LFWDataset
from trainer import fit
import numpy as np

cuda = torch.cuda.is_available()
import torch.backends.cudnn as cudnn
import argparse
import os
import collections
from PIL import Image

mean, std = 0.1307, 0.3081
# Training Settings
parser = argparse.ArgumentParser(description='PyTorch Face Re-Identification')

# Model options
parser.add_argument('--dataroot', type=str, default='./lfw-a', help='path to dataset')
parser.add_argument('--lfw-dir', type=str, default='./lfw-a', help='path to dataset')
parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt')
parser.add_argument('--log-dir', default='./pytorch_face_logs',
                    help='folder to output model checkpoints')
# Training options
parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=10000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=0.5, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
# train_dataset = MNIST('../data/MNIST', train=True, download=True,
#                              transform=transforms.Compose([
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((mean,), (std,))
#                              ]))
# test_dataset = MNIST('../data/MNIST', train=False, download=True,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((mean,), (std,))
#                             ]))
n_classes = 10
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
if args.cuda:
    cudnn.benchmark = True

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10' \
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin, args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']
#
# def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
#     plt.figure(figsize=(10,10))
#     for i in range(10):
#         inds = np.where(targets==i)[0]
#         plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     plt.legend(mnist_classes)
#
# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if cuda:
#                 images = images.cuda()
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labels

# triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
# triplet_test_dataset = TripletMNIST(test_dataset)
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
transform = transforms.Compose([
    Scale(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
triplet_train_dataset = TripletFaceDataset(dir=args.dataroot, n_triplets=args.n_triplets, transform=transform)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   **kwargs)

# triplet_train_dataset = TripletFaceDataset(
#     dir=args.dataroot,
#     n_triplets=args.n_triplets,
#     transform=transforms.Compose([
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((mean,), (std,))
#                              ])
# )

# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
# **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.lfw_dir, pairs_path=args.lfw_pairs_path,
               transform=transform), batch_size=batch_size, shuffle=False, **kwargs)


def main():
    # Set up the network and training parameters
    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if args.cuda or cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = create_optimizer(model, args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100
    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
