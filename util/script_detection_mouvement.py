# import necessary packages for hand gesture recognition project using Python OpenCV


import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model


class MediaPipeRecognizer:
    def __init__(self, model_path: str, class_names_path: str):
        # initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.model = load_model(model_path)
        with open(class_names_path) as f:
            self.class_names = f.read().split('\n')

    def recognize_frame(self, frame, sock, queue):
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

        # Get frame in RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = self.hands.process(frame_rgb)

        class_name = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for hands_lms in result.multi_hand_landmarks:
                for lm in hands_lms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                self.mpDraw.draw_landmarks(frame, hands_lms,
                                           self.mpHands.HAND_CONNECTIONS)

            # Predict gesture in Hand Gesture Recognition project
            prediction = self.model.predict([landmarks])
            class_id = np.argmax(prediction)
            class_name = self.class_names[class_id]

            # show the prediction on the frame
            cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        sock.SendData(class_name)
        queue.append(frame)
        #return class_name, frame
