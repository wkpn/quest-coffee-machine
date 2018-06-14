from settings import SHAPE_PREDICTOR_MODEL, FACE_RECOGNITION_MODEL, TRAIN_DATA_FOLDER, LABELS
import pickle
import dlib
import glob
import cv2
import os


def load_labels():
    with open(LABELS, 'rb') as labels_data:
        return pickle.load(labels_data)


def save_labels(labels):
    with open(LABELS, 'wb') as labels_data:
        pickle.dump(labels, labels_data)


def get_dlib_components():
    detector = dlib.get_frontal_face_detector()
    frm = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)

    return detector, frm, sp


def init_labels():
    detector, frm, sp = get_dlib_components()
    labels = {}

    for image_path in glob.glob(os.path.join(TRAIN_DATA_FOLDER, "*.jpg")):
        label = image_path[len(TRAIN_DATA_FOLDER) + 1:len(image_path) - 4]
        labels[label] = []

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        if len(dets) != 1:
            continue

        for _, d in enumerate(dets):
            shape = sp(gray, d)
            vector = frm.compute_face_descriptor(gray, shape)
            labels[label].append(vector)

    return labels


if __name__ == '__main__':
    print('Gathering information about labels from %s' % TRAIN_DATA_FOLDER)

    labels = init_labels()
    save_labels(labels)

    print('Done! Data saved to %s' % LABELS)
