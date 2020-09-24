# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
from imitations import load_imitations


def plot_acc(data1, name):
    acc1 = np.load(data1, allow_pickle = True)

    plt.figure()
    plt.plot(range(len(acc1)), acc1, label=name)

    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
    plt.savefig('images/'+str(name)+'.png')

# plot_acc('result/easy_gray.npy', 'MyNet')

def data_analyse_seven(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_break, right, right_break, acc, breaks, keep = 0,0,0,0,0,0,0
    for action in actions:
        if action[0] < 0 and action[2] == 0:       # left      
            left += 1
        elif action[0] < 0 and action[2] > 0:      # left break     
            left_break += 1
        elif action[0] > 0 and action[2] == 0:     # right      
            right += 1
        elif action[0] > 0 and action[2] > 0:      # right break     
            right_break += 1
        elif action[0] == 0 and action[1] > 0:     # accelerate
            acc += 1
        elif action[0] == 0 and action[2] > 0:     # break
            breaks += 1
        elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep
            keep += 1
    summ = left+left_break+right+right_break+acc+breaks+keep
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_break = ", left_break)
    print("Right = ", right)
    print("Right_break = ", right_break)
    print("Accelerate = ", acc)
    print("Break = ", breaks)
    print("Keep = ", keep)
    print("====================================================")

def data_analyse_four(data_folder):
    observations, actions = load_imitations(data_folder)
    left, right, acc, breaks = 0,0,0,0
    for action in actions:
        if action[0] < 0:         
            right += 1       # right
        elif action[0] > 0:
            left += 1        # left
        else:
            if action[1] > 0:
                acc += 1     # gas
            elif action[1] ==0 and action[2] > 0:
                breaks += 1  # brake
    summ = left+right+acc+breaks
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Right = ", right)
    print("Accelerate = ", acc)
    print("Break = ", breaks)
    print("====================================================")

def data_analyse_nine(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_break, left_break_gas, \
    right, right_break, right_break_gas, \
    acc, breaks, keep = 0,0,0,0,0,0,0,0,0

    for action in actions:
        if action[0] < 0 and action[1] > 0:
            left_break_gas+=1    # steering left and gas          
        elif action[0] < 0 and action[1] ==0 and action[2] ==0:
            left+=1              # steering left
        elif action[0] < 0 and action[1] ==0 and action[2] > 0:
            left_break+=1        # steering left and brake

        elif action[0] > 0 and action[1] > 0:
            right_break_gas+=1    # steering right and gas
        elif action[0] > 0 and action[1] ==0 and action[2] ==0:
            right+=1              # steering right
        elif action[0] > 0 and action[1] ==0 and action[2] > 0:
            right_break+=1        # steering right and brake

        elif action[0] == 0 and action[1] == 0 and action[2] == 0:
            keep+=1               # keep forward
        elif action[0] == 0 and action[2] > 0:
            breaks+=1             # brake
        elif action[0] == 0 and action[1] > 0 and action[2] == 0:
            acc+=1                # gas
    summ = left+left_break+left_break_gas+right+right_break+right_break_gas+acc+breaks+keep
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_break = ", left_break)
    print("Left_break_gas = ", left_break_gas)
    print("Right = ", right)
    print("Right_break_gas = ", right_break_gas)
    print("Left_break = ", left_break)
    print("Accelerate = ", acc)
    print("Break = ", breaks)
    print("Keep = ", keep)
    print("====================================================")

# data_analyse_four('data/teacher')
# data_analyse_seven('data/teacher')
# data_analyse_nine('data/teacher')
