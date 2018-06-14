from settings import SHAPE_PREDICTOR_MODEL, FACE_RECOGNITION_MODEL, TRAIN_DATA_FOLDER, LABELS, IMAGES, ANONYMOUS_UPN
from scipy.spatial import distance
import pickle
import glob
import dlib
import cv2
import os


class FaceRecognition:
    def __init__(self, unpickle=False):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)
        self.frm = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)
        self.train_data = TRAIN_DATA_FOLDER
        self.threshold = 0.55

        if unpickle:
            self.load_data()
        else:
            self.images = []
            self.labels = {}
            self.setup()

        self.alert_ready()

    @staticmethod
    def alert_ready():
        print('Recognizer is ready')

    def save_data(self):
        with open(LABELS, 'wb') as f:
            pickle.dump(self.labels, f)
        with open(IMAGES, 'wb') as f:
            pickle.dump(self.images, f)

    def load_data(self):
        with open(LABELS, 'rb') as f:
            self.labels = pickle.load(f)
        with open(IMAGES, 'rb') as f:
            self.images = pickle.load(f)

    def setup(self):
        self.load_images()
        self.load_vectors()

    def load_images(self):
        for image in glob.glob(os.path.join(self.train_data, "*.jpg")):
            label = image[len(self.train_data) + 1:len(image) - 4]
            self.images.append(image)
            self.labels[label] = []

    def load_vectors(self):
        for path in self.images:
            image = cv2.imread(path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray)

            if len(dets) != 1:
                continue

            for _, d in enumerate(dets):
                shape = self.sp(image, d)
                label = path[len(self.train_data) + 1:len(path) - 4]
                vector = self.frm.compute_face_descriptor(image, shape)
                self.labels[label].append(vector)

    def face_recognition(self, image):
        dets = self.detector(image)

        if len(dets) == 0:
            return [ANONYMOUS_UPN]
        else:
            result = []

            for _, d in enumerate(dets):
                shape = self.sp(image, d)
                face_descriptor = self.frm.compute_face_descriptor(image, shape)

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

            self.save_data()

            return result

    def recognize(self):
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        cap.release()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_recognition(gray)

        return result
