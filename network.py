# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import torch
import torch.nn as nn
import numpy as np

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        if torch.cuda.is_available(): 
            dev = "cuda:0" 
        else:  
            dev = "cpu"

        device = torch.device(dev)

        #=================================================================
        # EasyNet : 2 Conv2d and 2 Linear
        # 7 classes
        self.conv1 = nn.Conv2d(1, 16, 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3872, 128)
        self.fc2 = nn.Linear(128, 7)

        self.fc3 = nn.Linear(7, 4)
        #=================================================================
        # EasyNet : 2 Conv2d and 2 Linear
        # 1.2 (b) 4 classes
        self.fc22 = nn.Linear(128, 4)
        self.act_2 = nn.Sigmoid()
        
        #=================================================================
        # VGG16
        self.VGG16 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(4608, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 7),
        )
        #=================================================================
        self.Alexnet = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 192, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier_2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 7),
        )
        #=================================================================

    def normalization(self, x):
        """
        This funtion is used to normalize data based on the Max-Min method,
        the return value must between [-1, 1]
        x:        list of size len(x)
        return:   list of size len(x)
        """
        if len(x) > 1:
            x = ((x - min(x)) / (max(x) - min(x)))
        return x

    def extract_input(self, observation):
        """
        This funtion is used to normalize the sensor input,
        the return value must between [-1, 1]
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return:        list of 7
        """
        speed, abs_sensors, steering, gyroscope = \
        self.extract_sensor_values(observation, observation.shape[0])
        speed = self.normalization(speed)

        for i in range(4):
            abs_sensors[:,i] = self.normalization(abs_sensors[:,i])
        steering = self.normalization(steering)
        gyroscope = self.normalization(gyroscope)

        # print(speed.shape, abs_sensors.shape, steering.shape, gyroscope.shape)

        extract_input = torch.cat([speed, abs_sensors, steering, gyroscope], dim=1)

        return extract_input

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        #=================================================================
        #====================== Data Preprocessing =======================
        #=================================================================
        # Convert RGB to Grayscale
        rgb2gray = 0.2989*observation[:, :, :, 0] + \
                   0.5870*observation[:, :, :, 1] + \
                   0.1140*observation[:, :, :, 2]
            
        x = torch.reshape(rgb2gray, (-1, 96, 96, 1))
        x = x.permute(0, 3, 1, 2)
        #=================================================================
        # *** This part is used to add the sensor input ***
        #        uncommand with line 185 together
        # extract_input = self.extract_input(observation)
        # extract_input = self.fc3(extract_input)
        #=================================================================
        # EasyNet for 7 classes
        # x = observation.permute(0, 3, 1, 2)
        x = self.act(self.conv1(x)) 
        x = self.drop(x)
        x = self.act(self.conv2(x))
        x = self.drop(x)
        x = x.reshape(x.size(0), -1)
        # x = torch.cat([x, extract_input], dim=1)  # uncommand line 175,176 before
        x = self.fc1(x)
        x = self.fc2(x)
        #=================================================================
        # EasyNet for 4 classes
        # x = observation.permute(0, 3, 1, 2)
        # x = self.act(self.conv1(x)) 
        # x = self.drop(x)
        # x = self.act(self.conv2(x))
        # x = self.drop(x)
        # x = x.reshape(x.size(0), -1)
        # # x = torch.cat([x, extract_input], dim=1)  # uncommand line 175,176 before
        # x = self.fc1(x)
        # x = self.fc22(x)
        # x = self.act_2(x)
        #=================================================================
        # VGG16
        # x = observation.permute(0, 3, 1, 2)
        # x = self.VGG16(x)
        # h = x.reshape(x.shape[0], -1)
        # x = self.classifier(h)
        #=================================================================
        # AlexNet
        # x = observation.permute(0, 3, 1, 2)
        # x = self.Alexnet(x)
        # h = x.reshape(x.shape[0], -1)
        # x = self.classifier_2(h)
        #=================================================================
        return x

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        movement = []
        #=================================================================
        # Nine Classes Classification
        # for action in actions:
        #     if action[0] < 0 and action[1] > 0:
        #         movement.append(torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]))    # steering left and gas          
        #     elif action[0] < 0 and action[1] ==0 and action[2] ==0:
        #         movement.append(torch.Tensor([0, 1, 0, 0, 0, 0, 0, 0, 0]))    # steering left
        #     elif action[0] < 0 and action[1] ==0 and action[2] > 0:
        #         movement.append(torch.Tensor([0, 0, 1, 0, 0, 0, 0, 0, 0]))    # steering left and brake

        #     elif action[0] > 0 and action[1] > 0:
        #         movement.append(torch.Tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]))    # steering right and gas
        #     elif action[0] > 0 and action[1] ==0 and action[2] ==0:
        #         movement.append(torch.Tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]))    # steering right
        #     elif action[0] > 0 and action[1] ==0 and action[2] > 0:
        #         movement.append(torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0]))    # steering right and brake

        #     elif action[0] == 0 and action[1] == 0 and action[2] == 0:
        #         movement.append(torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0, 0]))    # keep forward
        #     elif action[0] == 0 and action[2] > 0:
        #         movement.append(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 0]))    # brake
        #     elif action[0] == 0 and action[1] > 0 and action[2] == 0:
        #         movement.append(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 1]))    # gas

        #=================================================================
        # Four Classes Classification
        # for action in actions:
        #     act = [0, 0, 0, 0]
        #     if action[0] < 0:         
        #         act[0] = 1        # left
        #     if action[0] > 0:
        #         act[1] = 1        # right
        #     if action[1] > 0:
        #         act[2] = 1        # gas
        #     if action[2] > 0:
        #         act[3] = 1        # brake
        #     movement.append(torch.Tensor(act))
        #=================================================================
        # Seven Classes Classification
        for action in actions:
            if action[0] < 0 and action[2] == 0:                         # left      
                movement.append(torch.Tensor([1, 0, 0, 0, 0, 0, 0]))
            elif action[0] < 0 and action[2] > 0:                        # left break     
                movement.append(torch.Tensor([0, 1, 0, 0, 0, 0, 0]))
            elif action[0] > 0 and action[2] == 0:                       # right      
                movement.append(torch.Tensor([0, 0, 1, 0, 0, 0, 0]))
            elif action[0] > 0 and action[2] > 0:                        # right break     
                movement.append(torch.Tensor([0, 0, 0, 1, 0, 0, 0]))   
            elif action[0] == 0 and action[1] > 0:                       # accelerate
                movement.append(torch.Tensor([0, 0, 0, 0, 1, 0, 0])) 
            elif action[0] == 0 and action[2] > 0:                       # break
                movement.append(torch.Tensor([0, 0, 0, 0, 0, 1, 0]))
            elif action[0] == 0 and action[1] == 0 and action[2] == 0:   # keep forward
                movement.append(torch.Tensor([0, 0, 0, 0, 0, 0, 1])) 
        #=================================================================
        return movement

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        # max_index = torch.argmax(scores, 1)
        # print(max_index)
        #=================================================================
        # Four Classes Classification
        # action = [0, 0, 0]
        # if scores[0][0] >= 0.5:
        #     action[0] = -1
        # if scores[0][1] >= 0.5:
        #     action[0] = 1
        # if scores[0][2] >= 0.5:
        #     action[1] = 1
        # if scores[0][3] >= 0.5:
        #     action[2] = 0.4

        #=================================================================
        # Nine Classes Classification
        # max_index = torch.argmax(scores, 1)
        # if max_index == 0:
        #     action = [-1.0, 1.0, 0]
        # elif max_index == 1:
        #     action = [-1.0, 0, 0]
        # elif max_index == 2:
        #     action = [-1.0, 0, 0.4]
        # elif max_index == 3:
        #     action = [1.0, 1.0, 0]
        # elif max_index == 4:
        #     action = [1.0, 0, 0]
        # elif max_index == 5:
        #     action = [1.0, 0, 0.4]
        # elif max_index == 6:
        #     action = [0, 0, 0]
        # elif max_index == 7:
        #     action = [0, 0, 0.4]
        # elif max_index == 8:
        #     action = [0, 1.0, 0]

        #=================================================================
        # Seven Classes Classification
        max_index = torch.argmax(scores, 1)
        if max_index == 0:
            action = [-1.0, 0.0, 0.0]
        elif max_index == 1:
            action = [-1.0, 0.0, 0.4]
        elif max_index == 2:
            action = [+1.0, 0.0, 0.0]
        elif max_index == 3:
            action = [+1.0, 0.0, 0.4]
        elif max_index == 4:
            action = [0.0, +1.0, 0.0]
        elif max_index == 5:
            action = [0.0, 0.0, 0.8]
        elif max_index == 6:
            action = [0.0, 0.0, 0.0]
        #=================================================================
        return (float(action[0]), float(action[1]), float(action[2]))

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
