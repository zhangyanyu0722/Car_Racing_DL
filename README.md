# Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network

<p align="center">
  <img src="https://github.com/zhangyanyu0722/Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network/blob/master/Figures/result.gif" height="300" width="600"/>
</p>

We present a mixed convolutional neural network to playing the CarRacing-v0 using imitation learning in OpenAI Gym. After training, this model can automatically detect the boundaries of road features and drive the robot like a human.

<br/>

**Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network**
<br/>
[Yanyu Zhang](https://yanyuzhang.com/) : zhangya@bu.edu
Hairuo Sun
<br/>

[Paper]
[Website](https://github.com/zhangyanyu0722/Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network)

## Prerequisites
```diff
! NOTE: this repo is made for PyTorch 1.0+ compatible issue, the generated results might be changed.
```

- Linux (tested on Ubuntu 16.04.4 LTS)
- Python 3.6.9
    - `3.6.4` tested
    - `3.6.9` tested
- PyTorch 1.3.1 
    - `1.0.0` (with CUDA 9.0, torchvision 0.2.1)
    - `1.1.0` (with CUDA 9.0, torchvision 0.3.0)
    - `1.3.1` (with CUDA 10.1, torchvision 0.4.2)
- nvcc 10.1
    - `9.0.176`, `10.1` compiling and execution tested
    - `9.2.88` execution only
- gcc 5.4.0
- Pyenv or Anaconda

and Python dependencies list in `3d-tracking/requirements.txt` 

## Quick Start
In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](3d-tracking/DOCUMENTATION.md).
