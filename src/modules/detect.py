import cv2

class Detect:
    def __init__(self):
        pass

    def detectImage(self):
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

    
    def detectVideo(self):
        # get video
        video = cv2.VideoCapture('src/assets/videos/traffic.mp4')
        # pre-trained car classifier
        classifier_file = 'src/lib/cars_detector.xml'

        # craete car classifier
        car_tracker = cv2.CascadeClassifier(classifier_file)

        while True:

            # Read the capture frame
            (read_successful, frame) = video.read()

            if read_successful:
                # must convert to grayscale
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                break

            # detect car
            cars = car_tracker.detectMultiScale(grayscale_frame)

            # Draw rectangles around the car
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 256, 0), 2)

            # display the image with the faces spotted
            cv2.imshow('WNM car detector', frame)

            # Dont autoclose (Wait here in the code and listen for a key press)
            key = cv2.waitKey(1)

            #### Stop if Q key is pressed
            if key == 18  or key == 113:
                break