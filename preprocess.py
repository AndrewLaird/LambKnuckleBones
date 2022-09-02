from PIL import Image
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters

# load the image


for dice in range(1, 7):
    # Open the image
    image = skimage.io.imread(f"dice/{dice}_reg.png")
    image_without_alpha = image[:, :, :3]

    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image_without_alpha)
    # create a mask based on the threshold
    t = 0.7
    binary_mask = gray_image > t
    # threshold it
    # masked_image =  gray_image > .1
    # print(masked_image)
    # save it as {dice}_thresholded
    plt.imshow(binary_mask, cmap="gray")
    plt.show()
    skimage.io.imsave(f"dice/{dice}_masked.png", binary_mask)
