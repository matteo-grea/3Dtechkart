import cv2
import time
import os


class SIFTRecognizer:
    def __init__(self, images_path, min_match_count, camera_index):
        self.camera_index = camera_index
        self.min_match_count = min_match_count
        self.images_names = []
        self.images = []
        for image in os.listdir(images_path):
            gray = cv2.cvtColor(cv2.imread(f"{images_path}/{image}"), cv2.COLOR_BGR2GRAY)
            self.images.append(gray)
            self.images_names.append(image[:-4])
        # use orb if sift is not installed
        self.feature_extractor = cv2.xfeatures2d.SIFT_create()

        # Get the images descriptors
        self.images_desc = []
        for i in self.images:
            kp, desc = self.feature_extractor.detectAndCompute(i, None)
            self.images_desc.append(desc)
        self.bf = cv2.BFMatcher()

    def recognize_camera_feed(self):
        cap = cv2.VideoCapture(self.camera_index)

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

            max_matches, idx_max = 0, -1
            for i, desc in enumerate(self.images_desc):
                kp_r, desc_r = self.feature_extractor.detectAndCompute(gray, None)
                matches = self.bf.knnMatch(desc, desc_r, k=2)

                # store all the good matches as per Lowe's ratio test.
                good_match = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_match.append(m)

                # if less than 150 points matched -> not the same images or higly distorted
                if len(good_match) > self.min_match_count and len(good_match) > max_matches:
                    max_matches = len(good_match)
                    idx_max = i
                    print(f"Good match for id: {i} with {len(good_match)} matches")

            best_match = self.images_names[idx_max] if idx_max != -1 else "No image matched the current frame"
            if idx_max != -1:
                yield best_match

            cv2.imshow("Ma caméra", frame)
            time.sleep(2)

            # Appuyez sur la touche 'q' pour quitter
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        # Libérer la capture vidéo et détruire toutes les fenêtres
        cap.release()
        cv2.destroyAllWindows()
