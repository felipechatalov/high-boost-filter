import cv2
import sys
import numpy as np


# 1920x1080 with some room to no crop at the top and bottom of the screen and proper aspect ratio
WIDTH_CAP = 1635
HEIGHT_CAP = 920

def try_resize_to_fit_screen(img):
    if img.shape[0] > WIDTH_CAP or img.shape[1] > HEIGHT_CAP:
        # Get the width and height proportions and subtract is value each
        # iteration until it fits the screen properly
        h_proportion = img.shape[0] / img.shape[1]
        w_proportion = img.shape[1] / img.shape[0]

        h = img.shape[0]
        w = img.shape[1]

        while w > WIDTH_CAP:
            w -= 1
            h -= h_proportion
        
        while h > HEIGHT_CAP:
            h -= 1
            w -= w_proportion
        
        h = int(h)
        w = int(w)

        print('Resizing image to fit screen: {}x{} -> {}x{}'.format(img.shape[0], img.shape[1], h, w))

        img = cv2.resize(img, (w, h))
    return img

def show_comparison(img, img2):
    # Concat both images side by side
    big_img = cv2.hconcat([img, img2])
    # Resize img to fit screen.
    # The image can go out of the screen bounds if it's too big,
    # cv opens image 1 to 1 in pixel ratio
    big_img = try_resize_to_fit_screen(big_img)

    cv2.imshow('Image', big_img)
    # Checks if any key was pressed or the 'X' button in the window was pressed
    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(100) > 0:
            break
    cv2.destroyAllWindows()

def highboost_filter(img, k):
    # Create kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1, k+8, -1],
        [-1, -1, -1]
    ])

    # Apply kernel
    img = cv2.filter2D(img, -1, kernel)

    return img

def main():
    # Read image path
    img_path = sys.argv[1]

    # Open image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # If not possible to open image
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    # Normalize img to 0-1
    img = img.astype(np.float32)/255

    # Apply highboost filter
    img2 = highboost_filter(img, 1)

    # Show the comparison on screen
    show_comparison(img, img2)

main()