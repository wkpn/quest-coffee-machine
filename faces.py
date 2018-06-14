from setup import load_data, save_data, get_dlib_components, load_images_and_vectors
from settings import ANONYMOUS_UPN
from scipy.spatial import distance
import cv2


class FaceRecognition:
    def __init__(self, unpickle=False):
        self.detector, self.frm, self.sp = get_dlib_components()
        self.threshold = 0.55

        if unpickle:
            self.labels, self.images = load_data()
        else:
            self.labels, self.images = load_images_and_vectors()

        self.alert_ready()

    @staticmethod
    def alert_ready():
        print('Recognizer is ready')

    def face_recognition(self):
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        cap.release()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.detector(gray)

        if len(dets) == 0:
            return [ANONYMOUS_UPN]
        else:
            result = []

            for _, d in enumerate(dets):
                shape = self.sp(gray, d)
                face_descriptor = self.frm.compute_face_descriptor(gray, shape)

                min_distance = 1000
                our_label = ''

                for labelDictKey in self.labels:
                    for vectorDict in self.labels[labelDictKey]:
                        calc_distance = distance.euclidean(vectorDict, face_descriptor)

                        if calc_distance < min_distance:
                            min_distance = calc_distance
                            our_label = labelDictKey

                if min_distance < self.threshold:
                    predict_name = our_label
                    result.append(predict_name)

                    self.labels[our_label].append(face_descriptor)

            save_data(self.labels, self.images)

            return result

