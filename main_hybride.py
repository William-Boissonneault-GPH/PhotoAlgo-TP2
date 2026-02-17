from imageio import imread
import skimage
from tp2_startcode.align_images import align_images
from tp2_startcode.crop_image import crop_image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale


###################Setup######################
###im1 = lowpassed
###im2 = highpassed

imageHybride_1 = {
    "input_im1_path": 'sourceImages\hybride\\albert_monroe\Marilyn_Monroe.png',
    "input_im2_path": 'sourceImages\hybride\\albert_monroe\Albert_Einstein.png',
    "nameOfResult": 'albert_monroe',
    "LowPass_cycle_per_img" : 7,
    "HighPass_cycle_per_img" : 11,
    "output_dir" : 'resultImages\\hybride',
    "alignementIsNeeded" : True,
    "Color" : False
}

###phase contrast + aDRiM 
### For fun, phase contrast for high frequency, aDRiM for low volumetric effect

imageHybride_2 = {
    "input_im1_path": 'sourceImages\hybride\PHC_aDRiM_superMicroscope\\aDRIM_bubble_resmatched_2.png',
    "input_im2_path": 'sourceImages\hybride\PHC_aDRiM_superMicroscope\PHC_bubble_resmatched.png',
    "nameOfResult": 'super_microscope',
    "LowPass_cycle_per_img" : 40,
    "HighPass_cycle_per_img" : 5,
    "output_dir" : 'resultImages\\hybride',
    "alignementIsNeeded" : True,
    "Color" : False
}


###meta morphose color, bain automne vers banc hiver
imageHybride_3 = {
    "input_im1_path": 'sourceImages\hybride\\albert_monroe\Marilyn_Monroe.png',
    "input_im2_path": 'sourceImages\hybride\\albert_monroe\Albert_Einstein.png',
    "nameOfResult": 'albert_monroe',
    "LowPass_cycle_per_img" : 7,
    "HighPass_cycle_per_img" : 11,
    "output_dir" : 'resultImages\\hybride',
    "alignementIsNeeded" : True,
    "Color" : False
}

### meta morphose black and white, vision vs polarimetric DOLP
imageHybride_4 = {
    "input_im1_path": 'sourceImages\hybride\\albert_monroe\Marilyn_Monroe.png',
    "input_im2_path": 'sourceImages\hybride\\albert_monroe\Albert_Einstein.png',
    "nameOfResult": 'albert_monroe',
    "LowPass_cycle_per_img" : 7,
    "HighPass_cycle_per_img" : 11,
    "output_dir" : 'resultImages\\hybride',
    "alignementIsNeeded" : True,
    "Color" : False
}
### meta morphose color, vision vs polarimetric AOLP
imageHybride_5 = {
    "input_im1_path": 'sourceImages\hybride\\albert_monroe\Marilyn_Monroe.png',
    "input_im2_path": 'sourceImages\hybride\\albert_monroe\Albert_Einstein.png',
    "nameOfResult": 'albert_monroe',
    "LowPass_cycle_per_img" : 7,
    "HighPass_cycle_per_img" : 11,
    "output_dir" : 'resultImages\\hybride',
    "alignementIsNeeded" : True,
    "Color" : False
}







ImageToHybrid = imageHybride_2
PresentationFigure = False
FrequencyFigure = True
contrast_stretched_allowed = False


def to_float01(im):
    im = im.astype(np.float32)
    # si image en uint8 -> 0..255, on met en 0..1
    if im.max() > 1.5:
        im /= 255.0
    return im

def sigma_from_cycles_per_image(cycles_per_image: float, H: int, W: int) -> float:
    """
    Vient de chat-GPT
    Convert cutoff in cycles/image to spatial Gaussian sigma (pixels),
    using -6 dB (|H|=0.5) cutoff definition and isotropic N=min(H,W).
    """
    N = min(H, W)
    fc = cycles_per_image / N  # cycles/pixel
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * fc)
    return float(sigma)


def fill_black_with_mean(img, threshold=0):
    """Remplace le noir par valeur moyenne image
    """

    # Work in float to avoid overflow
    img_float = img.astype(np.float32)
    non_black_mask = img_float > threshold

    # Compute mean of non-black pixels
    if np.any(non_black_mask):
        mean_val = np.mean(img_float[non_black_mask])
    else:
        mean_val = 0  # fallback if image is all black

    filled = img_float.copy()
    filled[~non_black_mask] = mean_val

    # Return in original dtype
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        filled = np.clip(filled, info.min, info.max)
        return filled.astype(img.dtype)

    return filled.astype(img.dtype)

def fill_black_local_mean(img, threshold=0, sigma=75, eps=1e-6):
    """
    Replace black pixels (<= threshold) with a local mean computed from nearby non-black pixels
    using normalized Gaussian convolution.

    img: 2D grayscale image (uint8/float)
    threshold: what counts as "black"
    sigma: neighborhood size (bigger = smoother fill)
    """
    img_f = img.astype(np.float32)

    # Mask: 1 where valid (non-black), 0 where black
    M = (img_f > threshold).astype(np.float32)

    # If everything is black, nothing to do
    if M.sum() < 1:
        return img.copy()

    # Weighted blur of values and blur of mask
    num = gaussian_filter(img_f * M, sigma=sigma, mode="nearest")
    den = gaussian_filter(M, sigma=sigma, mode="nearest")

    local_mean = num / np.maximum(den, eps)

    out = img_f.copy()
    out[M < 0.5] = local_mean[M < 0.5]

    # Cast back
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(out, info.min, info.max).astype(img.dtype)
    else:
        out = out.astype(img.dtype)

    return out



def hybrid_image_grayTone(im1, im2, cutoff_low=None, cutoff_high=None):
    # convert to float pour éviter overflow uint8
    # Dans fourrier: 
    # H = I1 G1 + I2 (1 - G2)
    # Dans espaces
    # H = i1 * g1 + i2 * (1-g2)

    ### J'utilise fréquences cycle/image comme dans l'article

    ###Assume niveau de gris
    im1 = to_float01(im1)
    im2 = to_float01(im2)

    #im1 = fill_black_local_mean(im1)
    #im2 = fill_black_local_mean(im2)


    ###Creation de G1 et G2
    sigma_g1 = sigma_from_cycles_per_image(cutoff_low, im1.shape[0], im1.shape[1])
    sigma_g2 = sigma_from_cycles_per_image(cutoff_low, im2.shape[0], im2.shape[1])

    #filtered img
    im1_f= skimage.filters.gaussian(im1, sigma_g1, channel_axis=None)
    im2_f= im2 - skimage.filters.gaussian(im2, sigma_g2, channel_axis=None)

    im12 = im1_f + 4 * im2_f


    ##contrast
    def contrast_stretch(img):
        img = img.astype(np.float32)
        imin = np.min(img)
        imax = np.max(img)

        out = (img - imin) / (imax - imin + 1e-8)

        return out

    if contrast_stretched_allowed:
        im12 = contrast_stretch(im12)

    if FrequencyFigure == True:
       
        def center_crop_2d(arr, crop_frac=0.5):
            """
            Center-crop a 2D array.
            crop_frac=0.5 keeps 50% of width/height (zoom in).
            """
            H, W = arr.shape[:2]
            ch = int(H * crop_frac)
            cw = int(W * crop_frac)
            y0 = (H - ch) // 2
            x0 = (W - cw) // 2
            return arr[y0:y0+ch, x0:x0+cw]

        def fft_log_mag(im):
            F = np.fft.fftshift(np.fft.fft2(im))
            mag = np.abs(F)
            return np.log1p(mag)

        def make_colored_spectra(im_low, im_high):
            # Compute log spectra
            S_low  = fft_log_mag(im_low)
            S_high = fft_log_mag(im_high)

            # Normalize both to [0,1] using same scale
            all_vals = np.concatenate([S_low.ravel(), S_high.ravel()])
            vmin, vmax = np.percentile(all_vals, (5, 99.5))

            S_low  = np.clip((S_low  - vmin) / (vmax - vmin + 1e-8), 0, 1)
            S_high = np.clip((S_high - vmin) / (vmax - vmin + 1e-8), 0, 1)

            # Create RGB images
            H, W = S_low.shape

            rgb_low  = np.zeros((H, W, 3))
            rgb_high = np.zeros((H, W, 3))
            rgb_mix  = np.zeros((H, W, 3))

            # Low-pass = RED
            rgb_low[..., 0] = S_low

            # High-pass = BLUE
            rgb_high[..., 2] = S_high

            # Hybrid = RED + BLUE
            rgb_mix[..., 0] = S_low
            rgb_mix[..., 2] = S_high

            return rgb_low, rgb_high, rgb_mix

        def plot_frequency_colored(im1_f, im2_f, im12):

            rgb_low, rgb_high, rgb_mix = make_colored_spectra(im1_f, im2_f)

            fig, axes = plt.subplots(2, 3, figsize=(12,8))

            # --- Spatial images ---
            axes[0,0].imshow(im1_f, cmap='gray', vmin=0, vmax=1)
            axes[0,0].set_title("Image (passe-bas)")

            axes[0,1].imshow(im2_f, cmap='gray')
            axes[0,1].set_title("Image (passe-haut)")

            axes[0,2].imshow(im12, cmap='gray', vmin=0, vmax=1)
            axes[0,2].set_title("Résultat hybride")

            # --- Frequency colored ---
            rgb_low = center_crop_2d(rgb_low)
            axes[1,0].imshow(rgb_low)
            axes[1,0].set_title("Spectre passe-bas (rouge)")

            rgb_high = center_crop_2d(rgb_high)
            axes[1,1].imshow(rgb_high)
            axes[1,1].set_title("Spectre passe-haut (bleu)")

            rgb_mix = center_crop_2d(rgb_mix)
            axes[1,2].imshow(rgb_mix)
            axes[1,2].set_title("Spectre global (rouge + bleu)")

            for ax in axes.ravel():
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(f'{ImageToHybrid["output_dir"]}\\{ImageToHybrid["nameOfResult"]}_frequency.png')
            plt.show()

        plot_frequency_colored(im1_f, im2_f, im12)

    return im12


# read images
im2 = imread(ImageToHybrid['input_im2_path'], pilmode='L')
im1 = imread(ImageToHybrid['input_im1_path'], pilmode='L')

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
if ImageToHybrid['alignementIsNeeded']:
    im1, im2 = align_images(im1, im2)

im1 = fill_black_local_mean(im1)
im2 = fill_black_local_mean(im2)

im1, im2= crop_image(im1, im2)

im12 = hybrid_image_grayTone(im1, im2, ImageToHybrid['LowPass_cycle_per_img'], ImageToHybrid['HighPass_cycle_per_img'])



def Figure_ingredients(im1,im2,im12):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=1)
    plt.title("image initial (BF)")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("image initial (HF)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(im12, cmap='gray', vmin=0, vmax=1)
    plt.title("Résultat hybridisation")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{ImageToHybrid["output_dir"]}\\{ImageToHybrid["nameOfResult"]}_ingredients.png')
    plt.show()


def Figure_three_views(img):
    fig = plt.figure(figsize=(6,4))

    # make 3 subplots with different physical widths
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.6, 0.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Proche")

    ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Moyenne")

    ax3.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("Éloignée")

    for ax in [ax1, ax2, ax3]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{ImageToHybrid["output_dir"]}\\{ImageToHybrid["nameOfResult"]}_visualsation.png')
    plt.show()


if PresentationFigure:
    Figure_ingredients(im1, im2, im12)
    Figure_three_views(im12)