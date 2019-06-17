import PIL
import scipy.misc
from io import BytesIO
import tensorboardX as tb
from tensorboardX.summary import Summary

class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def add_image(self, tag, img, step):
        summary = Summary()
        bio = BytesIO()

        if type(img) == str:
            img = PIL.Image.open(img)
        elif type(img) == PIL.Image.Image:
            pass
        else:
            img = PIL.Image.fromarray(img)

        img.save(bio, format="png")
        image_summary = Summary.Image(encoded_image_string=bio.getvalue())
        summary.value.add(tag=tag, image=image_summary)
        self.summary_writer.add_summary(summary, global_step=step)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)
