#!/usr/bin/env bash

echo "Downloading shape predictor model"
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
echo "Done!"

echo "Downloading face recognition model"
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
echo "Done!"
