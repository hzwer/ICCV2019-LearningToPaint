# Neural Stroke-based Rendering

![Image text](./image/step.png)

## Abstract

Excellent painters can use only a few strokes to create a fantastic painting, which is a symbol of human inte and art. Reversing the simulator to interpret images is also a challenging task of computer vision in recent years. In this paper, we propose a stroke-based rendering (SBR) method that combines the neural stroke renderer (NSR) and deep reinforcement learning (DRL), allowing the machine to learn the ability of deconstructing images using strokes and create amazing visual effects. Our agent is an end-to-end program that converts natural images into paintings. The training process does not require human painting experience or stroke tracking data. 

## Installation
Use [anaconda](https://conda.io/miniconda.html) to manage environment

```
$ conda create -n py36 python=3.6
$ source activate py36
```

### Dependencies
```
* [PyTorch](http://pytorch.org/) 0.4 
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
* [opencv-python](https://pypi.org/project/opencv-python/)
```

### Data
```
Download the CelebA(http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and put the align images in data/img_align_celeba/******.jpg
```

## Training

### NSR
To create a differentiable painting environment, we need train the NSR firstly. 

```
$ python3 train_bezier.py
$ tensorboard --logdir ./ --port=6006 
(The training process is shown in http://127.0.0.1:6006)
```

### RL agent
```
$ python3 train.py --max_step=40
```

## Results

![Image text](./image/CelebA.png)

![Image text](./image/imagenet.png)

## Reference

[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (model)
