import cv2

# get image
img_file = "src/assets/images/traffic.jpg"
# pre-trained car classifier
classifier_file = 'src/lib/cars_detector.xml'

# create opencv image
img = cv2.imread(img_file)
# must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# craete car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
# detect car
cars = car_tracker.detectMultiScale(grayscale_img)

# Draw rectangles around the car
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 256, 0), 2)


# display the image with the faces spotted
cv2.imshow('WNM car detector', img)

# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()