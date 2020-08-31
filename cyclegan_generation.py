import functools
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        content_image_path = sys.argv[1]
    else:
        sys.exit()
    parent_dir = str(Path(content_image_path).parent)

    if parent_dir != 'originals':
        print(f"${parent_dir} is not originals, exiting")
        sys.exit()
    # filepath = "originals/Everything_is_Going_to_be_Alright.png"

    style_image_path = Path("style_images")/np.random.choice(os.listdir("style_images"))

    # Load content and style images (see example in the attached colab).
    content_image = np.array(Image.open(content_image_path))
    style_image = np.array(Image.open(style_image_path))
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    style_image = tf.image.resize(style_image, (256, 256))

    # Load image stylization module.
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    Image.fromarray((stylized_image.numpy()[0] * 255 ).astype(np.uint8)).save(f'generated/{Path(content_image).stem}_modified.png')
