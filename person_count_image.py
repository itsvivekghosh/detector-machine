import cv2
import numpy as np
import imutils


def detect_person(image):
    proto_path = 'generic_models/MobileNetSSD_deploy.prototxt.txt'
    model_path = 'generic_models/MobileNetSSD_deploy.caffemodel'

    detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (w, h), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()

    print(person_detections.shape)
    no_of_persons = 0
    
    for i in np.arange(0, person_detections.shape[2]):

        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.25:
            no_of_persons += 1
            index = int(person_detections[0, 0, i, 1])

            if CLASSES[index] != 'person':
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, "{:.0f}% Person".format(confidence*100), (startX, startY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    print(no_of_persons)
    cv2.imshow('Persons', image)
    final_image = cv2.resize(image, (500, 500))
    cv2.imwrite("Persons_detected.jpg", final_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():

    image = cv2.imread('image2.jpg', 1)
    image = imutils.resize(image, width=600)

    detect_person(image)


if __name__ == '__main__':
    main()