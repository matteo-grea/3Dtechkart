from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os

liste_image = []
path_dir = "images"

for image in os.listdir(path_dir):
    rgb = cv2.cvtColor(cv2.imread(f"{path_dir}/{image}"), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    liste_image.append(gray)

# use orb if sift is not installed
feature_extractor = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with chosen feature_extractor
liste_desc = []
for i in liste_image :
    kp, desc = feature_extractor.detectAndCompute(i, None)
    liste_desc.append(desc)

bf = cv2.BFMatcher()
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



    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)   

    for i, desc in enumerate(liste_desc) :
        kp_r, desc_r = feature_extractor.detectAndCompute(gray, None)
        matches = bf.knnMatch(desc, desc_r, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_match = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_match.append(m)

        # if less then 10 points matched -> not the same images or higly distorted
        MIN_MATCH_COUNT = 100
        if len(good_match) > MIN_MATCH_COUNT:
            print('image detecte '+ str(i))



    cv2.imshow("Ma caméra", frame)
    time.sleep(2)

    # Appuyez sur la touche 'q' pour quitter
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et détruire toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
