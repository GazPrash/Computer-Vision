import cv2, numpy as np

def Convolver(image: np.ndarray, gfilter: np.ndarray):
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