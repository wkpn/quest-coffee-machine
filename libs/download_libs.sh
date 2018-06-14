#!/usr/bin/env bash

sudo apt-get update && sudo apt-get -y upgrade

sudo apt-get -y install libhdf5-dev libharfbuzz-dev libatlas-dev libatlas3-base libwebp-dev libtiff5-dev libjasper-dev \
libilmbase12 openexr libgst-dev gstreamer1.0-tools libavcodec-dev libavformat-dev libswscale-dev libqtgui4 libqt4-test \
python3-pip python3-rpi.gpio

pip3 install numpy h5py requests scipy opencv-contrib-python dlib

