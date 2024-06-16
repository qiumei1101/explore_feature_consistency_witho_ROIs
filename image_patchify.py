import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """ Load an image from the disk. """
    return Image.open(image_path)

def patchify(img, patch_size):
    """ Split the image into patches. """
    img_size = img.size
    patches = []
    for i in range(0, img_size[0], patch_size):
        for j in range(0, img_size[1], patch_size):
            patch = img.crop((i, j, i + patch_size, j + patch_size))
            patches.append(patch)
    return patches

def plot_patches(patches, num_cols):
    """ Plot the image patches. """
    num_rows = (len(patches) + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for ind, ax in enumerate(axs.flatten()):
        if ind < len(patches):
            ax.imshow(patches[ind], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()
    plt.savefig("pacthies.png")
    # plt.show()

# Example usage
image_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/0038_c017_00003340_0.jpg'  # Change to the path of your image
patch_size = 16  # Define the patch size, e.g., 16x16 pixels

# Load and patchify the image
img = load_image(image_path)
patches = patchify(img, patch_size)

# Plot the original image and the patches
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
# plt.show()

plot_patches(patches, num_cols=24)  # Adjust num_cols based on your preference
