import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def deskewImage(img):
    """
    adapted from: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    """
    
    image = img.copy()
    
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    print("Angle",angle)
    if angle > 45:
        angle = (90 - angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        return img

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_img = cv2.warpAffine(image, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_img

if __name__ == "__main__":
    # path to the image scan of the document
    
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename()

    # load the image from disk
    original_image = cv2.imread(file)
    deskewed_image = deskewImage(original_image)

    print("ORIGINAL IMAGE:")
    cv2.imshow("ORIGINAL IMAGE", original_image)
    cv2.waitKey(0)
    print()

    print("DESKEWED IMAGE:")
    cv2.imshow("DESKEWED IMAGE", deskewed_image)
    cv2.waitKey(0)
    