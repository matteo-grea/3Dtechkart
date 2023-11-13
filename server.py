# Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings

# Feel free to use this in your individual or commercial projects BUT make sure to reference me as: Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
# It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

# Use at your own risk
# Use under the Apache License 2.0

# Example of a Python UDP server

import UdpComms as U
import cv2
import threading
from util.script_detection_mouvement import MediaPipeRecognizer as MPR
from util.sript_detection_image import SIFTRecognizer

# Paramètres du modèle MediaPipe
model_path = "mp_hand_gesture"
class_names_path = "gesture.names"

# Charger le modèle MediaPipe
recognizer = MPR(model_path, class_names_path)

# Charger le reconnaisseur SIFT pour le changement de modèle
images_path = "images"
min_match_count = 150
sift_recognizer = SIFTRecognizer(images_path, min_match_count)


# Create UDP sockets to use for sending (and receiving), one sock for each script
sock_movement = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
sock_image = U.UdpComms(udpIP="127.0.0.1", portTX=8002, portRX=8003, enableRX=True, suppressWarnings=True)

# Get camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize a queue to get the frame to show
    queue = []

    # Initialize threads
    thread_movement = threading.Thread(target=recognizer.recognize_frame, args=(frame, sock_movement, queue,))
    thread_image = threading.Thread(target=sift_recognizer.recognize_frame, args=(frame, sock_image,))

    # Execute the recognize_frame methods from the models and send them to Unity
    thread_movement.start()
    thread_image.start()

    # Wait for both threads to finish executing
    thread_movement.join()
    thread_image.join()

    # Show the frame received from recognizer.recognize_frame stored in queue
    cv2.imshow("Output", queue[0])

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()