from faces import FaceRecognition
from settings import REST_API_URL
import RPi.GPIO as GPIO
import requests
import time


if __name__ == '__main__':
    channel = 17

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(channel, GPIO.IN)

    recognizer = FaceRecognition(unpickle=True)

    def callback(channel):
        result = recognizer.face_recognition()

        if not result:
            for upn in result:
                data = [{
                    "timestamp": int(time.time()),
                    "upn": upn,
                    "action": "Made a cup of coffee",
                    "message": "Thanks!"
                }]

                requests.post(REST_API_URL, json=data)


    GPIO.add_event_detect(channel, GPIO.BOTH, bouncetime=300)
    GPIO.add_event_callback(channel, callback)

    while True:
        time.sleep(1)
