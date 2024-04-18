from PIL import Image
import numpy as np
import imageutils as utils
import imageloader as loader

img = loader.load_image()
if img is not None:
    img_arr = np.array(img)
    sizes = tuple(img_arr.shape)
    flatten_img = img_arr.reshape(-1, img_arr.shape[-1])
    compressed_img = utils.compress_img_by_k(flatten_img, 10, 20)
    Image.fromarray(compressed_img.reshape(sizes)).show()