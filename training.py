# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import torch
import random
import time
from network import ClassificationNetwork
from imitations import load_imitations
import numpy as np

def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()

    infer_action.cuda()

    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-5) 
    # lr change from 1e-2 to 1e-5
    # Optimizer : Adam and SGD and RMSprop(For 4 classes)
    observations, actions = load_imitations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]
    gpu = torch.device('cuda')

    nr_epochs = 150
    batch_size = 64
    number_of_classes = 7  # needs to be changed  4/7/9
    start_time = time.time()
    loss_print = []   # add a list to record and print the loss later

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(gpu))
            batch_gt.append(batch[1].to(gpu, dtype=torch.int64))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))
                
                batch_out = infer_action(batch_in)
                # Three loss functions can be choise,
                loss = cross_entropy_loss(batch_out, batch_gt)
                # loss = binary_cross_entropy_loss(batch_out, batch_gt)
                # loss = RMSE_loss(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

        loss_print.append(total_loss)

    np.save('easy_gray.npy', loss_print)
    torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    criterion = torch.nn.CrossEntropyLoss()
    target = torch.argmax(batch_gt, 1)
    loss = criterion(batch_out, target)

    return loss

def binary_cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    # print(type(batch_out), type(batch_gt))
    criterion = torch.nn.BCELoss()
    loss = criterion(batch_out.type(torch.FloatTensor), batch_gt.type(torch.FloatTensor))

    return loss

def RMSE_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(x, y))

    return loss



