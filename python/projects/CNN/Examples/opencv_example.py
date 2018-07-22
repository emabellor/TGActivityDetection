import cv2 as cv

print('Trying to read image from file')
image = cv.imread('/home/mauricio/Pictures/429.jpg')

# Numpy array -> Checking type from elements
print('Checking type from variable')
print(str(type(image)))

cv.imshow('window', image)
print('Press a key to continue')
cv.waitKey(0)

cv.destroyWindow('window')

print('Trying to read from a fake route image')
image = cv.imread('/home/mauricio/Pictures/fakeImage.jpg')

if image is None:
    print('Cant read image')
else:
    cv.imshow('window2', image)
    print('Press a key again to continue')
    cv.waitKey(0)

    cv.destroyWindow('window2')


print('Done!')
print('OpenCV works with numpy array variables!')
