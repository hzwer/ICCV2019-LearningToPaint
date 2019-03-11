import cv2
import numpy as np

canvas_width = 256
output_width = 128

def normal(x):
    return (int)(x * (canvas_width - 1) + 0.5)

def draw(f):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0)
    x1 = normal(x1)
    x2 = normal(x2)
    y0 = normal(y0)
    y1 = normal(y1)
    y2 = normal(y2)
    z0 = (int)(1 + z0 * 64)
    z2 = (int)(1 + z2 * 64)
    canvas = np.zeros([canvas_width, canvas_width]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(output_width, output_width))
