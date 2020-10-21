# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img = cv2.imread("/Users/victoriaoberascher/Documents/FH_Hagenberg/5.Semester/ATF/Übungen/ATF_UE01_Zusatzmaterial/data/porsche.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)

    # show the images
    cv2.imshow("Original", img)
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
