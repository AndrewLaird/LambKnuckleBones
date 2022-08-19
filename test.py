from screenshot import process_image, students_t_test
import pyautogui
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


WIDTH = 2560
HEIGHT = 1440

im = pyautogui.screenshot(region=(0, 0, WIDTH, HEIGHT))

students_t_test(im)
