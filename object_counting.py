import numpy as np
from cv2 import cv2

# Load image - resize hình do kích thước hình quá to
image = cv2.imread('cell.jpg')
image = cv2.resize(image, (1200, 900))


# Convert to Gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Image binarization
ret ,binary_image = cv2.threshold(gray_image, 165, 255, cv2.THRESH_BINARY)


# Contour detection
_, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(image, contour, -1, (255, 0, 0), 3)

cv2.imshow('Image', image)
cv2.imshow('Gray image', gray_image)
cv2.imshow('Binary image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


