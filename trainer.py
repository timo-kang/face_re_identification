import torch
import numpy as np
from torch.nn import TripletMarginLoss
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import Function
from utils import PairwiseDistance

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def train_epoch_some(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    pbar = tqdm(enumerate(train_loader))
    labels, distances = [], []
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_p, data_n, label_p, label_n) in pbar:

        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        # Choose the hard negatives
        d_p = l2_dist.forward(out_a, out_p)
        d_n = l2_dist.forward(out_a, out_n)
        all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue
        out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
        out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
        out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())

        selected_data_a = Variable(torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).cuda())
        selected_data_p = Variable(torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).cuda())
        selected_data_n = Variable(torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).cuda())

        selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
        selected_label_n = torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
        triplet_loss = loss_fn.forward(out_selected_a, out_selected_p, out_selected_n)

        cls_a = model.forward_classifier(selected_data_a)
        cls_p = model.forward_classifier(selected_data_p)
        cls_n = model.forward_classifier(selected_data_n)

        cls_a = model.forward_classifier(selected_data_a)
        cls_p = model.forward_classifier(selected_data_p)
        cls_n = model.forward_classifier(selected_data_n)

        criterion = nn.CrossEntropyLoss()
        predicted_labels = torch.cat([cls_a, cls_p, cls_n])
        true_labels = torch.cat(
            [Variable(selected_label_p.cuda()), Variable(selected_label_p.cuda()), Variable(selected_label_n.cuda())])

        cross_entropy_loss = criterion(predicted_labels.cuda(), true_labels.cuda())

        loss = cross_entropy_loss + triplet_loss
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)