# Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings

# Feel free to use this in your individual or commercial projects BUT make sure to reference me as: Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
# It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

# Use at your own risk
# Use under the Apache License 2.0

# Example of a Python UDP server

import UdpComms as U
from util.script_detection_mouvement import MediaPipeRecognizer as MPR
import time
import cv2

model_path = "mp_hand_gesture"
class_names_path = "gesture.names"

# Charger le modèle MediaPipe
recognizer = MPR(model_path, class_names_path)

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

for class_name in recognizer.recognize_camera_feed():
    sock.SendData(class_name)
