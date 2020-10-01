# Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network

<p align="center">
  <img src="https://github.com/zhangyanyu0722/Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network/blob/master/Figures/result.gif" height="500" width="600"/>
</p>

We present a mixed convolutional neural network to playing the CarRacing-v0 using imitation learning in OpenAI Gym. After training, this model can automatically detect the boundaries of road features and drive the robot like a human.

<br/>

**Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network**
<br/>
[Yanyu Zhang](https://yanyuzhang.com/) : zhangya@bu.edu

Hairuo Sun : hrsun@bu.edu
<br/>

[Paper]
[Website](https://github.com/zhangyanyu0722/Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network)

## Prerequisites
```diff
! NOTE: this repo is made for PyTorch 1.6 compatible issue, the generated results might be changed.
```
- Linux (tested on Ubuntu 16.04.4 LTS)
- Python 3.6.9
    - `3.6.9` tested
- PyTorch 1.6.0
    - `1.6.0` (with CUDA 10.2, torchvision 0.7.0)
- Anaconda

and Python dependencies list in `requirements.txt` 

## Quick Start
In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.

### Installation
- Clone this repo:
```bash
git clone https://github.com/zhangyanyu0722/Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network.git
cd Deep-Reinforcement-Learning-with-Mixed-Convolutional-Network/
```

- Install PyTorch 1.6.0 and torchvision 0.7.0 from http://pytorch.org and other dependencies.
```bash
# Install gym (recommand conda)
conda install -c conda-forge gym
# Or
pip install 'gym[all]'
```
- Install requirements.
```bash
pip3 install -r requirements.txt
```
<p align="center">
Model | Score
-----|------
[Best Model](https://drive.google.com/file/d/1g4oiER4ZFwLVu1ssUcbQifj4iBeNf13M/view?usp=sharing) | 649.1
[EasyNet_RGB](https://drive.google.com/file/d/1GLK9af4OUU8GmmNMiOh61pmKWKhCqWmH/view?usp=sharing) | 438.8
[EasyNet_Gray](https://drive.google.com/file/d/1a63waR8AA-yNFJ8FjUKkhu0cvYXjb7VU/view?usp=sharing) | 432.4
[AlexNet_RGB](https://drive.google.com/file/d/17L2ZqE12jmdBLMrPzQEOWPDvcDAU8q9h/view?usp=sharing) | 471.5
[AlexNet_Gray](https://drive.google.com/file/d/17n-Zf5HyKIYqP9Vh95UrbIYXIblEz8a4/view?usp=sharing) | 464.9
[VGG16_RGB](https://drive.google.com/file/d/1npkvXvTZvkxhyx7EIRlzGEc5L8U5I_r3/view?usp=sharing) | 594.9
[VGG16_Gray](https://drive.google.com/file/d/1xsCawTvq3nVlHreO2e8IqXa7XzhoxL7L/view?usp=sharing) | 558.3
</p>



