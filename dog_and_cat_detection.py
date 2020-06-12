import numpy as np
import cv2
import imutils


class DetectDogCat:

    def __init__(self):
        self.type, self.image_path, self.video_path = "image", None, None
        self.prop_txt_path = 'generic_models/MobileNetSSD_deploy.prototxt.txt'
        self.model_path = "generic_models/MobileNetSSD_deploy.caffemodel"

    def preProcessData(self, path):

        self.image_path = path
        image = cv2.imread(self.image_path, 1)
        # image = cv2.resize(image, (500, 500))
        image = imutils.resize(image, width=600)
        w, h = image.shape[:2]

        return image, w, h

    def detectDogCatFromModel(self, path):

        detector = cv2.dnn.readNetFromCaffe(prototxt=self.prop_txt_path, caffeModel=self.model_path)

        CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        image, w, h = self.preProcessData(path)

        blob = cv2.dnn.blobFromImage(image, 0.007843, (w, h), 127.5)
        detector.setInput(blob)
        detect_Animals = detector.forward()

        for i in np.arange(0, detect_Animals.shape[2]):

            confidence = detect_Animals[0, 0, i, 2]

            index = int(detect_Animals[0, 0, i, 1])
            if CLASSES[index] == 'dog' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Dog".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            elif CLASSES[index] == 'cat' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Cat".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            elif CLASSES[index] == 'horse' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Horse".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            elif CLASSES[index] == 'bird' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Bird".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            elif CLASSES[index] == 'sheep' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Sheep".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            elif CLASSES[index] == 'cow' and confidence >= 0.5:
                box = detect_Animals[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "{:.1f} Cow".format(confidence * 100), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

            cv2.imshow("image", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect = DetectDogCat()

    detect.detectDogCatFromModel(path='images/xrainbowlorikeet-2.jpg.pagespeed.ic.ePoxQcNwjc.jpg')
    detect.detectDogCatFromModel(path='images/Can-cats-and-dogs-get-coronavirus_resized.jpg')
