from settings import SHAPE_PREDICTOR_MODEL, FACE_RECOGNITION_MODEL, TRAIN_DATA_FOLDER, LABELS, IMAGES
import pickle
import dlib
import glob
import cv2
import os


def load_data():
    with open(LABELS, 'rb') as labels_data, open(IMAGES, 'rb') as images_data:
        labels = pickle.load(labels_data)
        images = pickle.load(images_data)

        return labels, images


def save_data(labels, images):
    with open(LABELS, 'wb') as labels_data, open(IMAGES, 'wb') as images_data:
        pickle.dump(labels, labels_data)
        pickle.dump(images, images_data)


def get_dlib_components():
    detector = dlib.get_frontal_face_detector()
    frm = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)

    return detector, frm, sp


def load_images_and_vectors():
    images = []
    labels = {}

    for image in glob.glob(os.path.join(TRAIN_DATA_FOLDER, "*.jpg")):
        label = image[len(TRAIN_DATA_FOLDER) + 1:len(image) - 4]
        images.append(image)
        labels[label] = []

    detector, frm, sp = get_dlib_components()

    for image_path in images:
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        if len(dets) != 1:
            continue

        for _, d in enumerate(dets):
            shape = sp(image, d)
            label = image_path[len(TRAIN_DATA_FOLDER) + 1:len(image_path) - 4]
            vector = frm.compute_face_descriptor(image, shape)
            labels[label].append(vector)

    return labels, images


if __name__ == '__main__':
    print('Gathering information about labels and images from %s' % TRAIN_DATA_FOLDER)

    labels, images = load_images_and_vectors()
    save_data(labels, images)

    print('Done! Data saved to %s and %s' % (LABELS, IMAGES))
