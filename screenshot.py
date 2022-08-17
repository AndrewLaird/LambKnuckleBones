import pyautogui
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import numpy as np


MASTER_WIDTH = 2560
MASTER_HEIGHT = 1440

# CHANGE THESE GUYS TO YOUR RESOLUTION
WIDTH = 2560
HEIGHT = 1440

RATIO_X = WIDTH / MASTER_WIDTH
RATIO_Y = HEIGHT / MASTER_HEIGHT

BOX_TOP_LEFT_X = int(150 * RATIO_X)
BOX_TOP_LEFT_Y = int(1020 * RATIO_Y)
BOX_BOTTOM_RIGHT_X = int(590 * RATIO_X)
BOX_BOTTOM_RIGHT_Y = int(1230 * RATIO_Y)

DICE_WIDTH = int(130 * RATIO_X)
DICE_HEIGHT = int(130 * RATIO_Y)

im = pyautogui.screenshot(region=(0, 0, WIDTH, HEIGHT))


rolling_box = im.crop(
    box=(BOX_TOP_LEFT_X, BOX_TOP_LEFT_Y, BOX_BOTTOM_RIGHT_X, BOX_BOTTOM_RIGHT_Y)
)

# convert the image to grayscale
gray_rolling_box = skimage.color.rgb2gray(rolling_box)

plt.imshow(gray_rolling_box, cmap="gray")
plt.show()

print(gray_rolling_box.shape)


def get_top_left_of_die(gray_rolling_box):
    for y, row in enumerate(gray_rolling_box):
        for x, col in enumerate(row):
            if col > 0.5:
                return (x - 5 * RATIO_X, y)
    return (0, 0)


dice_top_x, dice_top_y = get_top_left_of_die(gray_rolling_box)

def inverse(value):
    return 1-value

def compare_to_dice(gray_dice, die_num: int):
    die_img = Image.open(f"dice/{die_num}_masked.png")
    # match gray_dice shape
    die_img = die_img.resize((DICE_WIDTH, DICE_HEIGHT))
    die_img = process_image(die_img)

    die_img_np = np.array(die_img)
    gray_dice_np = np.array(gray_dice)
    
    # inverse our images
    die_img_np_inverse = np.array(list(map(inverse,die_img_np)))
    gray_dice_np_inverse = np.array(list(map(inverse,gray_dice_np)))

    
    difference_die = die_img_np_inverse - gray_dice_np_inverse
    difference_die = np.array(list(map(lambda x: abs(x),difference_die)))

    similarity = np.sum(difference_die)
    return similarity



def process_image(img):
    gray_image = skimage.color.rgb2gray(img)
    t = 0.7
    binary_mask = gray_image > t
    return binary_mask



def check_dice_from_top_left(dice_top_x, dice_top_y):
    dice_image = im.crop(
        box=(dice_top_x, dice_top_y, dice_top_x + DICE_WIDTH, dice_top_y + DICE_HEIGHT)
    )

    gray_dice= process_image(dice_image)
    best_difference = float('inf')
    best_difference_num = 1
    for die_num in range(1,7):
        difference = compare_to_dice(gray_dice, die_num)
        if(difference < best_difference):
            best_difference_num = die_num
            best_difference = difference
    return best_difference_num


check_dice_from_top_left(dice_top_x + BOX_TOP_LEFT_X, dice_top_y + BOX_TOP_LEFT_Y)
