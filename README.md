# ICCV2019-Learning to Paint

## [arXiv](https://arxiv.org/abs/1903.04411) | [YouTube](https://youtu.be/YmOgKZ5oipk) | [Reddit](https://www.reddit.com/r/reinforcementlearning/comments/b5lpfl/learning_to_paint_with_modelbased_deep/) | [Slide(中文)](https://docs.google.com/presentation/d/1itHk_yI8847wx-meH9k0v_8dNZS2dD0p/edit?usp=sharing&ouid=101528516762521089540&rtpof=true&sd=true) | [DeepWiki](https://deepwiki.com/hzwer/ICCV2019-LearningToPaint) | [Replicate](https://replicate.ai/hzwer/iccv2019-learningtopaint)
[Zhewei Huang](https://scholar.google.com/citations?user=zJEkaG8AAAAJ&hl=zh-CN&oi=sra), Wen Heng, [Shuchang Zhou](https://scholar.google.com/citations?user=zYI0rysAAAAJ&hl=zh-CN&oi=sra)

## Abstract

We show how to teach machines to paint like human painters, who can use a
small number of strokes to create fantastic paintings. By employing a neural
renderer in model-based Deep Reinforcement Learning (DRL), our agents learn to
determine the position and color of each stroke and make long-term plans to
decompose texture-rich images into strokes. Experiments demonstrate that
excellent visual effects can be achieved using hundreds of strokes. The
training process does not require the experience of human painters or stroke
tracking data. 

**You can easily use [colaboratory](https://colab.research.google.com/github/hzwer/LearningToPaint/blob/master/LearningToPaint.ipynb) to have a try.**

![Demo](./demo/lisa.gif)![Demo](./demo/sunrise.gif)![Demo](./demo/sunflower.gif)
![Demo](./demo/palacemuseum.gif)![Demo](./demo/deepdream_night.gif)![Demo](./demo/deepdream_bird.gif)

### Dependencies
* [PyTorch](http://pytorch.org/) 1.1.0 
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
* [opencv-python](https://pypi.org/project/opencv-python/) 3.4.0
```
pip3 install torch==1.1.0
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
$ ffmpeg -r 10 -f image2 -i output/generated%d.png -s 512x512 -c:v libx264 -pix_fmt yuv420p video.mp4 -q:v 0 -q:a 0
(make a painting process video)
```

We also provide with some other neural renderers and agents, you can use them instead of renderer.pkl to train the agent:

[triangle.pkl](https://drive.google.com/open?id=1YefdnTuKlvowCCo1zxHTwVJ2GlBme_eE) --- [actor_triangle.pkl](https://drive.google.com/open?id=1k8cgh3tF7hKFk-IOZrgsUwlTVE3CbcPF);

[round.pkl](https://drive.google.com/open?id=1kI4yXQ7IrNTfjFs2VL7IBBL_JJwkW6rl) --- [actor_round.pkl](https://drive.google.com/open?id=1ewDErUhPeGsEcH8E5a2QAcUBECeaUTZe);

[bezierwotrans.pkl](https://drive.google.com/open?id=1XUdti00mPRh1-1iU66Uqg4qyMKk4OL19) --- [actor_notrans.pkl](https://drive.google.com/open?id=1VBtesw2rHmYu2AeJ22XvTCuzuqkY8hZh)

We also provide 百度网盘 source. 链接: https://pan.baidu.com/s/1GELBQCeYojPOBZIwGOKNmA 提取码: aq8n 
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
$ cd baseline
$ python3 train.py --max_step=40 --debug --batch_size=96
(A step contains 5 strokes in default.)
$ tensorboard --logdir train_log --port=6006
```

## Resources
[量子位报道](https://zhuanlan.zhihu.com/p/64097633)

[Learning to Paint：一个绘画 AI](https://zhuanlan.zhihu.com/p/61761901)

[旷视研究院推出基于深度强化学习的绘画智能体](https://zhuanlan.zhihu.com/p/80732065)

* Our ICCV poster
  <div>
  <img src="./image/poster.png" width="800">
  </div>
* [Our ICCV rebuttal for reviewers](https://drive.google.com/file/d/1bEBS-uxmVEc7WVuX35NCodxDu17s_d8m/view?usp=sharing)
## Contributors
- [hzwer](https://github.com/hzwer)
- [ak9250](https://github.com/ak9250)

Also many thanks to [ctmakro](https://github.com/ctmakro/rl-painter) for inspiring this work. He also explored using greedy algorithm to generate paintings - [opencv_playground](https://github.com/ctmakro/opencv_playground).

If you find this repository useful for your research, please cite the following paper:
```
@inproceedings{huang2019learning,
  title={Learning to paint with model-based deep reinforcement learning},
  author={Huang, Zhewei and Heng, Wen and Zhou, Shuchang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
