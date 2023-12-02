import cv2
import PIL
import math
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
GAUSSIAN_KERNEL_2D = lambda x, y, sigma_sqr: (
    (math.e) ** -(0.5 * ((x ** 2 + y ** 2) / sigma_sqr))
) / (2 * math.pi * sigma_sqr)


def Convolution(image: np.ndarray, gfilter: np.ndarray):
    for col in range(gfilter.shape[0]):
        gfilter[col, :] = gfilter[gfilter.shape[0] -1 - col, :]

    for row in range(gfilter.shape[1]):
        gfilter[:, row] = gfilter[:, gfilter.shape[1] -1 - row]


    ResultMatrix = np.zeros(image.shape)
    filterCenter = np.array(gfilter.shape) // 2

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            # SingleConvolve(i, j, )
            totalSum = 0
            for k in range(gfilter.shape[1]):
                for l in range(gfilter.shape[0]):
                    if ((i - filterCenter[1]) < 0) or (
                        (i + filterCenter[1] >= image.shape[1]) or
                        (j - filterCenter[0] < 0) or 
                        (j + filterCenter[0] >= image.shape[0])
                    ):
                        pixelValue = 0
                    else:
                        pixelValue = image[j - filterCenter[0] + l, i - filterCenter[1] + k] * gfilter[l, k]
                        
                    totalSum += pixelValue

            ResultMatrix[j, i][:] = totalSum

    return ResultMatrix


def GaussianBlur(shape: tuple, variance: int = 4):
    # shape should be col, row
    # filter_size = 2 * int(4 * variance + 0.5) + 1
    mask = np.zeros(shape, np.float32)
    m = shape[1] // 2
    n = shape[0] // 2

    for rind in range(-m, m + 1):
        for cind in range(-n, n + 1):
            mask[cind + n, rind + m] = GAUSSIAN_KERNEL_2D(rind, cind, variance ** 2)

    return mask


path = "images/smallimg1.png"
imgmat = cv2.imread(path)
greymat = cv2.cvtColor(imgmat, cv2.COLOR_BGR2GRAY)

blur = GaussianBlur((9, 9), 8)

convmatrix = Convolution(imgmat, blur)

cv2.imwrite("images/gblureed_sm1.png", convmatrix)