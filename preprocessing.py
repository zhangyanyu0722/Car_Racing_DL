# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import numpy as np
from analyse import data_analyse_seven
from imitations import load_imitations
import random
import os

def data_reduce(data_folder):
    """
    Reduce data to balance the distribution
    """
    observations, actions = load_imitations(data_folder)
    select = []
    for index in range(len(actions)):
        if actions[index][0] == 0 and actions[index][1] > 0:     # accelerate
            select.append(index)
    random.shuffle(select)
    select = select[:int(len(select)/2)]
    for i in select:
        delete(data_folder + '/action_'+'%05d' % i + '.npy')
        delete(data_folder + '/observation_'+'%05d' % i + '.npy')
    print("========= Done!, " + str(int(len(select))) + \
          " Acc have been deleted ========")
    print("====================================================")


def delete(data_address):
    """
    Delete files from OS
    """
    os.remove(data_address)

if __name__ == "__main__":
    print("Be careful before running this file")
    if len(sys.argv) == 1 or sys.argv[1] == "prep":
        data_analyse_seven('data/teacher')
        data_reduce('data/teacher')
        data_analyse_seven('data/teacher')
    else:
        print('This command is not supported')


