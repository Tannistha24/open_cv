import cv2 as cv
import numpy as np
import urllib.request
import os

# Haar cascade for face detection
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
base_url = "https://raw.githubusercontent.com/jasmcaus/opencv-course/master/Resources/Faces/train"

features, labels = [], []

def create_train():
    for person in people:
        label = people.index(person)
        for i in range(1, 6):  # assuming 5 images per person
            url = f"{base_url}/{person}/{i}.jpg"
            try:
                resp = urllib.request.urlopen(url)
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv.imdecode(img_array, cv.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)
            except:
                continue

create_train()


print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
