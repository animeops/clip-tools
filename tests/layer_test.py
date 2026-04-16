from clip_tools import ClipImage
from PIL import Image
import numpy as np

clip_image = ClipImage.open("tests/test_data/test000.clip")

clip_image_composited = clip_image.composite()

clip_image_gt = Image.open("tests/test_data/test000.png")
clip_image_gt = np.array(clip_image_gt)

def print_layers(layers, prefix=""):
    for layer in layers:
        print(prefix + layer.name)
        if layer.is_group():
            print_layers(layer, prefix + "--")

print_layers(clip_image)

assert np.array_equal(np.array(clip_image_composited), clip_image_gt)