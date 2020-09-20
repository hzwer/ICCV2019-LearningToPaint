import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.stroke_gen import *

writer = TensorBoard("../train_log/")
import torch.optim as optim
import copy
import os
import datetime
from datetime import datetime, date
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import tensorflow as tf
import tensorflow_hub as hub

# import spiral.agents.default as default_agent
# import spiral.agents.utils as agent_utils
#from spiral.environments import fluid
from spiral.environments import libmypaint

from numpy import save

nest = tf.contrib.framework.nest

# Disable TensorFlow debug output.
tf.logging.set_verbosity(tf.logging.ERROR)


criterion = nn.MSELoss()
net = FCN()
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 64

use_cuda = torch.cuda.is_available()
step = 0


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "./newModel/renderer.pkl")
    if use_cuda:
        net.cuda()


def load_weights():
    pretrained_dict = torch.load("./renderer.pkl")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


# The path to libmypaint brushes.
#BRUSHES_BASEDIR = os.path.join("/content/spiral/spiral/third_party/mypaint-brushes-1.3.0")
BRUSHES_BASEDIR = os.path.abspath("/content/spiral/third_party/mypaint-brushes-1.3.0")
print(BRUSHES_BASEDIR)
# The path to a TF-Hub module.
MODULE_PATH = "https://tfhub.dev/deepmind/spiral/default-wgangp-celebahq64-gen-19steps/agent4/1"

env_settings = dict(
    episode_length=20,                 # Number of frames in each episode.
    canvas_width=128,                   # The width of the canvas in pixels.
    grid_width=32,                     # The width of the action grid.
    brush_type="classic/dry_brush",    # The type of the brush.
    brush_sizes=[1, 2, 4, 8, 12, 24],  # The sizes of the brush to use.
    use_color=True,                    # Color or black & white output?
    use_pressure=True,                 # Use pressure parameter of the brush?
    use_alpha=False,                   # Drop or keep the alpha channel of the canvas?
    background="white",                # Background could either be "white" or "transparent".
    brushes_basedir=BRUSHES_BASEDIR,   # The location of libmypaint brushes.
)
env = libmypaint.LibMyPaint(**env_settings)
# print(env.step(None))
time_step = env.reset()
load_weights()
time = datetime.now().time()
while step < 100000:
    net.train()
    train_batch = []
    ground_truth = []

    
    for i in range(batch_size):
      #reset enviornment 
        env.reset()

      #create new action
        flag = np.random.randint(2)
        control = np.random.randint(1024)
        end = np.random.randint(1024)
        size = np.random.randint(6)
        blue = np.random.randint(20)
        green = np.random.randint(20)
        red = np.random.randint(20)
        pressure = np.random.randint(10)
        f = np.random.uniform(0, 1, 8)
        f[0] = control
        f[1] = end
        f[2] = flag
        f[3] = pressure
        f[4] = size
        f[5] = red
        f[6] = green
        f[7] = blue
      #alpha = np.random.randint(10)
      #speed = np.random.randint(9)

        action = {"control": np.array(control,dtype='int32'),
              "end": np.array(end,dtype='int32'),
              "flag": np.array(flag,dtype='int32'),
              "pressure": np.array(pressure,dtype='int32'),
              "size": np.array(size,dtype='int32'),
              "red": np.array(red,dtype='int32'),
              "green": np.array(green,dtype='int32'),
              "blue": np.array(blue,dtype='int32')}
    #apply action
        time_step = env.step(action)
    #append action and canvas to the lists        
        train_batch.append(f)
    #print(time_step.observation["canvas"].shape)
        ground_truth.append(time_step.observation["canvas"])

    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    from numpy import save
    #print(gen)
    #save('trial' +'.npy', gen.detach().numpy())
    #print(ground_truth)
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    
    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 100 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)

        # for i in range(32):
        #     G = gen[i].cpu().data.numpy()
        #     GT = ground_truth[i].cpu().data.numpy()
        #     writer.add_image("train/gen{}.png".format(i), G, step)
        #     writer.add_image("train/ground_truth{}.png".format(i), GT, step)
    if step % 1000 == 0:
        new_time = datetime.now().time()
        print("spentime:" , datetime.combine(date.today(), new_time) - datetime.combine(date.today(), time))
        time= new_time
        save('./newFiles/GT' +str(step) +'.npy', ground_truth.cpu().detach().numpy())
        save('./newFiles/generate' +str(step) +'.npy', gen.cpu().detach().numpy())
        print(step, loss.item())
        save_model()
    step += 1
