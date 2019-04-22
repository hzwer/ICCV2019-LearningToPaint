# Learning to Paint with Model-based Deep Reinforcement Learning

---> https://arxiv.org/abs/1903.04411

## Abstract

We show how to teach machines to paint like human painters, who can use a few strokes to create fantastic paintings. By combining the neural renderer and model-based Deep Reinforcement Learning (DRL), our agent can decompose texture-rich images into strokes and make long-term plans. For each stroke, the agent directly determines the position and color of the stroke. Excellent visual effect can be achieved using hundreds of strokes. The training process does not require experience of human painting or stroke tracking data. 

You can easily use [colaboratory](https://colab.research.google.com/github/hzwer/LearningToPaint/blob/master/LearningToPaint.ipynb) to have a try.

![Demo](./demo/lisa.gif)![Demo](./demo/sunrise.gif)![Demo](./demo/sunflower.gif)

[**YouTube demo**](https://www.youtube.com/watch?v=UqE5cytdrdc)

[**bilibili demo**](https://www.bilibili.com/video/av50005448)

**Mona Lisa** using different renderers and spliced lisa 

![Lisa](./image/lisa.png) ![Lisa0](./image/lisa_0.png) ![Lisa1](./image/lisa_1.png) ![Lisa2](./image/lisa_2.png) ![Lisa3](./image/lisa_3.png)

![Architecture](./image/main.png)

## Installation
Use [anaconda](https://conda.io/miniconda.html) to manage environment

```
$ conda create -n py36 python=3.6
$ source activate py36
$ git clone https://github.com/hzwer/LearningToPaint.git
$ cd LearningToPaint
```

### Dependencies
* [PyTorch](http://pytorch.org/) 0.4.1 
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
* [opencv-python](https://pypi.org/project/opencv-python/) 3.4.0
```
pip3 install opencv-python==0.4.1
pip3 install tensorboardX
pip3 install opencv-python
```

## Testing
Make sure there are renderer.pkl and actor.pkl before testing.

You can download a trained neural renderer and a CelebA actor for test: [renderer.pkl](https://drive.google.com/open?id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4) and [actor.pkl](https://drive.google.com/open?id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR)

```
$ wget "https://drive.google.com/uc?export=download&id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4" -O renderer.pkl
$ wget "https://drive.google.com/uc?export=download&id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR" -O actor.pkl
$ python3 baseline/test.py --max_step=100 --actor=actor.pkl --renderer=renderer.pkl --img=image/test.png --divide=4
$ ffmpeg -r 30 -f image2 -i output/generated%d.png -s 512x512 -vcodec mpeg4 video.mp4 -q:v 0 -q:a 0
(make a painting process video)
```

We also provide with some other neural renderers and agents, you can use them instead of renderer.pkl to train the agent:

[triangle.pkl](https://drive.google.com/open?id=1YefdnTuKlvowCCo1zxHTwVJ2GlBme_eE) --- [actor_triangle.pkl](https://drive.google.com/open?id=1k8cgh3tF7hKFk-IOZrgsUwlTVE3CbcPF);

[round.pkl](https://drive.google.com/open?id=1kI4yXQ7IrNTfjFs2VL7IBBL_JJwkW6rl) --- [actor_round.pkl](https://drive.google.com/open?id=1ewDErUhPeGsEcH8E5a2QAcUBECeaUTZe);

[bezierwotrans.pkl](https://drive.google.com/open?id=1XUdti00mPRh1-1iU66Uqg4qyMKk4OL19)

## Training

### Datasets
Download the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and put the aligned images in data/img_align_celeba/\*\*\*\*\*\*.jpg

### Neural Renderer
To create a differentiable painting environment, we need train the neural renderer firstly. 

```
$ python3 baseline/train_renderer.py
$ tensorboard --logdir train_log --port=6006
(The training process will be shown at http://127.0.0.1:6006)
```

### Paint Agent
After the neural renderer looks good enough, we can begin training the agent.
```
$ python3 baseline/train.py --max_step=200 --debug --batch_size=96
(A step contains 5 strokes in default.)
$ tensorboard --logdir train_log --port=6006
```
## Results

<div align=center>
<img src="./image/step.png" width="500">
</div>

![Results](./image/results.png)

If you find this repository useful for your research, please cite the following paper:

```
@article{huang2019learning,
  title={Learning to Paint with Model-based Deep Reinforcement Learning},
  author={Huang, Zhewei and Heng, Wen and Zhou, Shuchang},
  journal={arXiv preprint arXiv:1903.04411},
  year={2019}
}
```

## Resources
- A Chinese introduction [Learning to Paint：一个绘画 AI](https://zhuanlan.zhihu.com/p/61761901)
- A Chinese tutorial [[教程]三分钟学会画一个爱豆](https://zhuanlan.zhihu.com/p/63194822)


## Contributors
- [hzwer](https://github.com/hzwer)
- [ak9250](https://github.com/ak9250)
