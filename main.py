import cv2
image = cv2.imread("Sree.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(height, width) = gray_image.shape[:2]
center = (width //2, height //2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(gray_image, matrix, (width, height))
cv2.imshow("Rotated Gray Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("rotated_Sree.jpg",rotated_image)