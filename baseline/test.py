import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
args = parser.parse_args()

T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) # Coordconv

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer))

def decode(x, canvas): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas[0])
    return canvas, res

img = cv2.imread(args.img, cv2.IMREAD_COLOR)
img = cv2.resize(img, (width, width))
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img).to(device).reshape(1, -1, width, width).float() / 255.
actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)
output = canvas[0].detach().cpu().numpy()
output = np.transpose(output, (1, 2, 0))

os.system('mkdir output')
cv2.imwrite('output/generated0.png', (output * 255).astype('uint8'))

with torch.no_grad():
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res = decode(actions, canvas)
        print('step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
        for j in range(5):
            output = res[j].detach().cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            cv2.imwrite('output/generated' + str(i * 5 + j + 1) + '.png', (output * 255).astype('uint8'))
