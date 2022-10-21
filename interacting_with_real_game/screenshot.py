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

x_positions = [975, 1218, 1455]
y_positions = [
    104,
    254,
    404,
    900,
    1050,
    1200,
]
die_positions = [[[x_positions[x], y_positions[y]] for x in range(3)] for y in range(6)]


def get_top_left_of_die(gray_rolling_box):
    for y, row in enumerate(gray_rolling_box):
        for x, col in enumerate(row):
            if col > 0.5:
                return (x - 5 * RATIO_X, y)
    return (0, 0)


def inverse(value):
    return 1 - value


def compare_to_dice(gray_dice, die_num: int):
    die_img = Image.open(f"dice/{die_num}_masked.png")
    # match gray_dice shape
    die_img = die_img.resize((DICE_WIDTH, DICE_HEIGHT))
    die_img = process_image(die_img)

    die_img_np = np.array(die_img)
    gray_dice_np = np.array(gray_dice)

    # inverse our images
    die_img_np_inverse = np.array(list(map(inverse, die_img_np)))
    gray_dice_np_inverse = np.array(list(map(inverse, gray_dice_np)))

    difference_die = die_img_np_inverse - gray_dice_np_inverse
    difference_die = np.array(list(map(lambda x: abs(x), difference_die)))

    similarity = np.sum(difference_die)
    return similarity


def process_image(img):
    gray_image = skimage.color.rgb2gray(img)
    t = 0.3
    binary_mask = gray_image > t
    return binary_mask


def check_dice_from_top_left(dice_top_x, dice_top_y, screenshot):
    dice_image = screenshot.crop(
        box=(dice_top_x, dice_top_y, dice_top_x + DICE_WIDTH, dice_top_y + DICE_HEIGHT)
    )
    # plt.imshow(dice_image, cmap="gray")
    # plt.show()

    gray_dice = process_image(dice_image)
    best_difference = float("inf")
    best_difference_num = 1
    for die_num in range(0, 7):
        difference = compare_to_dice(gray_dice, die_num)
        if difference < best_difference:
            best_difference_num = die_num
            best_difference = difference
    return best_difference_num


def students_t_test(screenshot):
    top_left = [750, 678]
    bottom_right = [810, 753]

    T_on_file = Image.open(f"dice/T.png")
    T_on_screen = screenshot.crop(
        box=(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    )
    T_on_file = process_image(np.array(T_on_file)[:, :, :3])
    T_on_screen = process_image(np.array(T_on_screen)[:, :, :3])
    # difference

    T_on_file_np = np.array(T_on_file)
    T_on_screen_np = np.array(T_on_screen)

    # inverse our images
    T_on_file_np_inverse = np.array(list(map(inverse, T_on_file_np)))
    T_on_screen_np_inverse = np.array(list(map(inverse, T_on_screen_np)))

    difference_t = T_on_file_np_inverse - T_on_screen_np_inverse
    difference_t = np.array(list(map(lambda x: abs(x), difference_t)))
    # plt.imshow(difference_t, cmap="gray")
    # plt.show()

    similarity = np.sum(difference_t)
    return similarity < 500


def click_board(action: int):
    # click outside of the boxes
    # they start us in the middle
    if action < 1:
        pyautogui.press("a")
    elif action > 1:
        pyautogui.press("d")
    pyautogui.press("e")


def read_board():
    im = pyautogui.screenshot(region=(0, 0, WIDTH, HEIGHT))
    rolling_box = im.crop(
        box=(BOX_TOP_LEFT_X, BOX_TOP_LEFT_Y, BOX_BOTTOM_RIGHT_X, BOX_BOTTOM_RIGHT_Y)
    )
    rolling_box = process_image(rolling_box)
    dice_top_x, dice_top_y = get_top_left_of_die(rolling_box)

    rolled_num = check_dice_from_top_left(
        dice_top_x + BOX_TOP_LEFT_X, dice_top_y + BOX_TOP_LEFT_Y, im
    )
    board = [[0 for i in range(3)] for j in range(6)]
    for row, die_row in enumerate(die_positions):
        for col, die_col in enumerate(die_row):
            board[row][col] = check_dice_from_top_left(die_col[0], die_col[1], im)

    # we currently have 6 rows of 3
    # we want two board with 3 column
    # and the bottom one flipped
    final_board = [
        [[0 for row in range(3)] for col in range(3)] for player_board in range(2)
    ]

    for row in range(6):
        for col in range(3):
            player = row <= 2
            final_row = row % 3
            if player:
                # flip that shit
                final_row = 2 - final_row

            final_board[player][col][final_row] = board[row][col]
    game_over = False
    winner = 0
    game_over, winner = read_game_over(im)

    return final_board, rolled_num, game_over, winner


PLUS_PIXELS = [1185, 768]
END_GAME_BANNER_PIXELS = [1120, 750]


def read_game_over(screenshot):
    game_over = screenshot.getpixel(
        (END_GAME_BANNER_PIXELS[0], END_GAME_BANNER_PIXELS[1])
    ) == (0, 0, 0)
    winner = students_t_test(screenshot)
    return game_over, winner
