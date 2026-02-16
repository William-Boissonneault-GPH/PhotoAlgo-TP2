import glob
import os
import numpy as np
import skimage
from imageio import imread
from tp1_startcode.tp1_io import quantize_to_8bit
import matplotlib.pyplot as plt

"""ce code principal prend toutes les images du input dir et applique du sharppening, fait une figure de comparaison du niveau de sharpening utilisé et recrache des images du niveau moyen dans output direct

Crée aussi le template d'une section de rapport"""

images_a_accentuer_input_dir = 'sourceImages\\accentuation'
images_a_accentuer_output_dir = ''

extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]

#va chercher tout les chemins d'image
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(images_a_accentuer_input_dir, ext)))

#load les imgs
images_a_accentuer = []

for img_path in image_paths:
    images_a_accentuer.append(imread(img_path))

def applySharpening(image, spatial_sigma, alpha):
    print(image.shape)
    float_img = image.astype(np.float32) / 255.0
    gaussian_filtered = skimage.filters.gaussian(float_img, sigma=spatial_sigma, channel_axis=-1)
    sharpenned_image = float_img + alpha * (float_img - gaussian_filtered)
    return quantize_to_8bit(sharpenned_image)

newTestImage = applySharpening(images_a_accentuer[2], 6, 0.5)
plt.imshow(newTestImage)
plt.show()


