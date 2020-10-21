# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img = cv2.imread("dice.png", cv2.IMREAD_COLOR)

    edges = cv2.Canny(img, 200, 100);

    cv2.imshow("demo", edges)
    cv2.waitKey(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
