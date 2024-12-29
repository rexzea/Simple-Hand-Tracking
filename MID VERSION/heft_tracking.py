import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot
import json
from datetime import datetime

class AdvancedHandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        self.p_time = 0
        self.c_time = 0

        self.gestures = {
            "palm": False,
            "fist": False,
            "pointing": False,
            "pinch": False,
            "peace": False
        }
        
        # ini warna
        self.colors = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }

    def find_hands(self, img, draw=True, flip=True):
        if flip:
            img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        self.all_hands = []
        h, w, c = img.shape
        
        if self.results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                hand_info = {}
                landmarks = []
                
                for lm_id, lm in enumerate(hand_landmarks.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                    landmarks.append([px, py, pz])
                
                hand_info["landmarks"] = landmarks
                hand_info["type"] = "Right" if self.results.multi_handedness[hand_id].classification[0].label == "Right" else "Left"
                self.all_hands.append(hand_info)
                
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )

                    cv2.putText(img, hand_info["type"], 
                              (landmarks[0][0]-20, landmarks[0][1]-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["blue"], 2)
        
        return img

    def find_gesture(self, img):
        for hand in self.all_hands:
            landmarks = hand["landmarks"]
            
            if len(landmarks) == 21:
                self.gestures = dict.fromkeys(self.gestures, False)

                fingers = self.count_fingers(landmarks)

                if sum(fingers) == 5:
                    self.gestures["palm"] = True
                    gesture_name = "Palm"

                elif sum(fingers) == 0:
                    self.gestures["fist"] = True
                    gesture_name = "Fist"

                elif fingers == [0, 1, 0, 0, 0]:
                    self.gestures["pointing"] = True
                    gesture_name = "Pointing"

                elif fingers == [0, 1, 1, 0, 0]:
                    self.gestures["peace"] = True
                    gesture_name = "Peace"

                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                distance = hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
                
                if distance < 40:
                    self.gestures["pinch"] = True
                    gesture_name = "Pinch"
                else:
                    gesture_name = "Unknown"

                cv2.putText(img, f"Gesture: {gesture_name}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, self.colors["green"], 2)
        
        return img

    def count_fingers(self, landmarks):
        fingers = []
        
        if landmarks[4][0] > landmarks[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        for tip in [8, 12, 16, 20]:
            if landmarks[tip][1] < landmarks[tip-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def calculate_fps(self, img):
        self.c_time = time.time()
        fps = 1 / (self.c_time - self.p_time)
        self.p_time = self.c_time
        
        cv2.putText(img, f"FPS: {int(fps)}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, self.colors["green"], 2)
        return img

    def save_gesture_data(self):
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gestures": self.gestures
        }
        
        try:
            with open("gesture_log.json", "a") as f:
                json.dump(data, f)
                f.write("\n")
        except Exception as e:
            print(f"Error saving gesture data: {e}")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = AdvancedHandTracker(detection_con=0.8, track_con=0.8)

    cv2.namedWindow("Advanced Hand Tracking", cv2.WINDOW_NORMAL)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        img = tracker.find_hands(img)
        img = tracker.find_gesture(img)
        img = tracker.calculate_fps(img)

        tracker.save_gesture_data()

        cv2.imshow("Advanced Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()