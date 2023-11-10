# Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings

# Feel free to use this in your individual or commercial projects BUT make sure to reference me as: Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
# It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

# Use at your own risk
# Use under the Apache License 2.0

# Example of a Python UDP server

import UdpComms as U
import time
import numpy as np
import cv2

model_file = 'MobileNetSSD_deploy.caffemodel'
config_file = 'MobileNetSSD_deploy.prototxt'

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

# Charger le modèle Caffe
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

CLASSES = ["arriere-plan", "avion", "velo", "oiseau", "bateau",
           "bouteille", "autobus", "voiture", "chat", "chaise", "vache", "table",
           "chien", "cheval", "moto", "personne", "plante en pot", "mouton",
           "sofa", "train", "moniteur"]

# Ouvrir la capture vidéo à partir de la première caméra (index 0)
cap = cv2.VideoCapture(0)

# Vérifier si la capture vidéo est ouverte
if not cap.isOpened():
    print("Erreur: La caméra n'est pas disponible")
    exit()

while True:
    # Lire un cadre vidéo
    ret, frame = cap.read()

    # Vérifier si la lecture s'est bien déroulée
    if not ret:
        print("Erreur: Impossible de lire la trame vidéo")
        break

    # Prétraitement de l'image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Définir l'entrée du réseau
    net.setInput(blob)

    # Effectuer la détection d'objet
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]
            #box = detections[0, 0, i, 3:7] * np.array(
            #    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            #(startX, startY, endX, endY) = box.astype(int)

            # Dessiner le rectangle autour de l'objet détecté
            #color = (0, 255, 0)
            #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Afficher le label de l'objet
            #cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            sock.SendData(label)

    cv2.imshow("Ma caméra", frame)
    time.sleep(1)

    # Appuyez sur la touche 'Esc' ('Echap') pour quitter
    if cv2.waitKey(1000) == 27:
        break


# Libérer la capture vidéo et détruire toutes les fenêtres
cap.release()
cv2.destroyAllWindows()