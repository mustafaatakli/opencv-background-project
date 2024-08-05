import cv2
import numpy as np

image = cv2.imread('image.jpeg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

edges = cv2.Canny(img_blur, 50, 150)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_image = np.zeros_like(image)
areas = [916, 1732, 1334]

for contour in contours:
    alan = cv2.contourArea(contour)
    if int(alan) in areas:
        cv2.drawContours(result_image, [contour], -1, (0, 0, 0), thickness=1)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    else:
        cv2.drawContours(result_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

cv2.imshow('Resim_Cikti', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()