import cv2
import numpy as np
import imutils


class DetectPerson:

    def __init__(self, prop_path, model_path, video_path):
        self.prop_path = prop_path
        self.model_path = model_path
        self.video_path = video_path

    def detect_person_from_video(self):

        detector = cv2.dnn.readNetFromCaffe(prototxt=self.prop_path, caffeModel=self.model_path)

        CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        # video = cv2.VideoCapture(self.video_path)
        video = cv2.VideoCapture(0)

        while True:
            ret, frame = video.read()
            frame = cv2.resize(frame, (500, 500))
            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.001, (w, h), 100)

            detector.setInput(blob)
            person_detections = detector.forward()

            for i in np.arange(0, person_detections.shape[2]):

                confidence = person_detections[0, 0, i, 2]

                if confidence > 0.5:
                    index = int(person_detections[0, 0, i, 1])

                    if CLASSES[index] != 'person':
                        continue

                    person_box = person_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = person_box.astype("int")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, "{:.1f} Person".format(confidence*100),
                        (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), 2
                    )

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    obj = DetectPerson(
        prop_path='generic_models/MobileNetSSD_deploy.prototxt.txt',
        model_path='generic_models/MobileNetSSD_deploy.caffemodel',
        video_path='video/mask.mp4'
    )

    obj.detect_person_from_video()
