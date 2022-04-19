import argparse
import tempfile
from pathlib import Path

import cog
import cv2
import imageio

from baseline.DRL.actor import *
from baseline.Renderer.model import *


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser = argparse.ArgumentParser(description="Learning to Paint")
        parser.add_argument(
            "--max_step", default=80, type=int, help="max length for episode"
        )
        parser.add_argument(
            "--imgid", default=0, type=int, help="set begin number for generated image"
        )
        parser.add_argument(
            "--divide",
            default=3,
            type=int,
            help="divide the target image to get better resolution",
        )
        self.args = parser.parse_args("")

    @cog.input("image", type=Path, help="input image")
    @cog.input(
        "renderer",
        type=str,
        default="default",
        options=["default", "triangle", "round", "bezierwotrans"],
        help="type of renderer",
    )
    def predict(self, image, renderer="default"):
        width = 128
        self.args.max_step = 80
        all_images = []
        self.args.img = str(image)
        self.args.actor = "actors/actor_" + renderer + ".pkl"
        self.args.renderer = "renderers/" + renderer + ".pkl"

        canvas_cnt = self.args.divide * self.args.divide
        T = torch.ones([1, 1, width, width], dtype=torch.float32).to(self.device)
        img = cv2.imread(self.args.img, cv2.IMREAD_COLOR)
        origin_shape = (img.shape[1], img.shape[0])

        coord = torch.zeros([1, 2, width, width])
        for i in range(width):
            for j in range(width):
                coord[0, 0, i, j] = i / (width - 1.0)
                coord[0, 1, i, j] = j / (width - 1.0)
        coord = coord.to(self.device)  # Coordconv

        Decoder = FCN()
        Decoder.load_state_dict(torch.load(self.args.renderer))

        actor = ResNet(9, 18, 65)  # action_bundle = 5, 65 = 5 * 13
        actor.load_state_dict(torch.load(self.args.actor))
        actor = actor.to(self.device).eval()
        Decoder = Decoder.to(self.device).eval()

        canvas = torch.zeros([1, 3, width, width]).to(self.device)

        patch_img = cv2.resize(
            img, (width * self.args.divide, width * self.args.divide)
        )
        patch_img = large2small(patch_img, canvas_cnt, self.args, width)
        patch_img = np.transpose(patch_img, (0, 3, 1, 2))
        patch_img = torch.tensor(patch_img).to(self.device).float() / 255.0

        img = cv2.resize(img, (width, width))
        img = img.reshape(1, width, width, 3)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.tensor(img).to(self.device).float() / 255.0

        with torch.no_grad():
            if self.args.divide != 1:
                self.args.max_step = self.args.max_step // 2
            for i in range(self.args.max_step):
                stepnum = T * i / self.args.max_step
                actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
                canvas, res = decode(actions, canvas, Decoder, width)
                print(
                    "canvas step {}, L2Loss = {}".format(
                        i, ((canvas - img) ** 2).mean()
                    )
                )
                for j in range(5):
                    img_j = save_img(res[j], origin_shape, self.args, width)
                    all_images.append(img_j)
                    self.args.imgid += 1
            if self.args.divide != 1:
                canvas = canvas[0].detach().cpu().numpy()
                canvas = np.transpose(canvas, (1, 2, 0))
                canvas = cv2.resize(
                    canvas, (width * self.args.divide, width * self.args.divide)
                )
                canvas = large2small(canvas, canvas_cnt, self.args, width)
                canvas = np.transpose(canvas, (0, 3, 1, 2))
                canvas = torch.tensor(canvas).to(self.device).float()
                coord = coord.expand(canvas_cnt, 2, width, width)
                T = T.expand(canvas_cnt, 1, width, width)
                for i in range(self.args.max_step):
                    stepnum = T * i / self.args.max_step
                    actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                    canvas, res = decode(actions, canvas, Decoder, width)
                    print(
                        "divided canvas step {}, L2Loss = {}".format(
                            i, ((canvas - patch_img) ** 2).mean()
                        )
                    )
                    for j in range(5):
                        img_j = save_img(res[j], origin_shape, self.args, width, True)
                        all_images.append(img_j)
                        self.args.imgid += 1

        out_path = Path(tempfile.mkdtemp()) / "out.gif"

        print("generating gif")
        imageio.mimwrite(str(out_path), all_images, duration=0.02)
        return out_path


def decode(x, canvas, Decoder, width):  # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res


def small2large(x, args, width):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x


def large2small(x, canvas_cnt, args, width):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x


def smooth(img, args, width):
    def smooth_pix(img, tx, ty):
        if (
            tx == args.divide * width - 1
            or ty == args.divide * width - 1
            or tx == 0
            or ty == 0
        ):
            return img
        img[tx, ty] = (
            img[tx, ty]
            + img[tx + 1, ty]
            + img[tx, ty + 1]
            + img[tx - 1, ty]
            + img[tx, ty - 1]
            + img[tx + 1, ty - 1]
            + img[tx - 1, ty + 1]
            + img[tx - 1, ty - 1]
            + img[tx + 1, ty + 1]
        ) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def save_img(res, origin_shape, args, width, divide=False):
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output, args, width)
        output = smooth(output, args, width)
    else:
        output = output[0]
    output = (output * 255).astype("uint8")
    output = cv2.resize(output, origin_shape)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output
