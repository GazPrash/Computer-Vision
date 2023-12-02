import cv2


image = cv2.imread("images/2.jpg")
image = cv2.resize(image, (round(image.shape[0] * 0.5), round(image.shape[1] * 0.5)))

d = [7, 9, 15, 30]
sigmas = [(10, 10), (30, 30), (75, 75), (100, 100), (125, 125), (150, 150)]

# for i in d:
#     for j in sigmas:
#         bilateimg = cv2.bilateralFilter(image, i, *j)
#         print(i, j)
#         cv2.imshow("", bilateimg)
#         cv2.waitKey(5 * 1000)

bilateimg = cv2.bilateralFilter(image, d[2], *(sigmas[-4]))
cv2.imshow("", bilateimg)
cv2.waitKey(5 * 1000)

# performs well from 15 (75) to 30 (125)
