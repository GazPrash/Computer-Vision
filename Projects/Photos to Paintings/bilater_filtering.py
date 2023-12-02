import cv2
import numpy as np
from convolutions import Convolver


IntensityKernel = lambda k, sigma_i : (np.exp(-(k**2)/2*sigma_i**2))
# intensity_kernel = np.exp(-(image_gray[i-d:i+d+1, j-d:j+d+1] - image_gray[i,j])**2 / (2 * sigma_i**2))
SpatialKernel = lambda x, y, sigma_s : (np.exp(-(x**2 + y**2)/(2 * sigma_s**2)))

def BilateralFilter(image:np.ndarray, sigma_spatial:int, sigma_freq:int, d:int):
    PaintedImage = np.zeros(image.shape)

    # # spatial coords of the image
    # These arrays can then be used to calculate the Euclidean distance between each pixel 
    # in the local window and the central pixel, which is used to compute the spatial weights in the bilateral filter.
    x, y = np.meshgrid(np.arange(-d, d+1), np.arange(-d, d+1))

    for i in range(d, image.shape[1] - d):
        for j in range(d, image.shape[0] - d):
            k = image[j-d:j+d+1, i-d:i+d+1][:] - image[j, i][:]
            # spatial = np.stack([SpatialKernel(i-x, j-y, sigma_spatial)] * 3, axis=-1)
            Kernel = np.stack([SpatialKernel(i-x, j-y, sigma_spatial)] * 3, axis=-1) * IntensityKernel(k, sigma_freq)
            # NormKernel = np.linalg.norm(Kernel) # normalizing kernel
            NormKernel = Kernel / np.sum()

            PaintedImage[j, i][:] = np.sum(image[j, i][:] * NormKernel)

        print(f"{i=}")

    return PaintedImage


if __name__ == "__main__":
    image = cv2.imread("images/alan.jpg")
    image = cv2.resize(image, (int(image.shape[0] * 0.4), int(image.shape[0] * 0.4)))
    bfImage = BilateralFilter(image, 75, 75, 15)
    cv2.imwrite("images/BFimg.png", bfImage)
