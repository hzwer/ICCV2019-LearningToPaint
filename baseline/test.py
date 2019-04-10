import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.stroke_gen import *
from DRL.ddpg import decode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--path', default='./model/Paint-run1/', type=str, help='Actor model path')
args = parser.parse_args()

T = torch.ones([1, 1, width, width], dtype=torch.float32)

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) # Coordconv

img = cv2.imread('./image/test.png', cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (width, width))
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img).to(device).reshape(1, -1, width, width).float() / 255.
actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.path + '/actor.pkl'))
actor = actor.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)

for i in range(args.max_step):
    stepnum = T * i / args.max_step
    actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
    canvas = decode(actions, canvas)
    print('step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))

output = canvas[0].detach().numpy()
output = np.transpose(output, (1, 2, 0))
cv2.imwrite('./generated.png', (output * 255).astype('uint8'))
