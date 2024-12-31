import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

class RexzeaHandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.gestures = {
            'fist': self.is_fist,
            'palm': self.is_palm,
            'victory': self.is_victory,
            'thumbs up': self.is_thumbs_up
        }

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
        return landmark_list

    def get_distance(self, p1, p2):
        return sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def is_fist(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        palm_center = landmarks[0]
        
        distances = [
            self.get_distance(palm_center, tip) 
            for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        ]
        avg_distance = sum(distances) / len(distances)
        
        return avg_distance < 100 

    def is_palm(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False
            
        tips = [landmarks[i] for i in [8, 12, 16, 20]]  
        mcp = [landmarks[i] for i in [5, 9, 13, 17]]   
        
        fingers_open = all(tip[2] < mcp[i][2] for i, tip in enumerate(tips))
        
        return fingers_open

    def is_victory(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False

        index_tip = landmarks[8][2]
        middle_tip = landmarks[12][2]
        ring_tip = landmarks[16][2]
        pinky_tip = landmarks[20][2]
        
        index_pip = landmarks[6][2]
        middle_pip = landmarks[10][2]
        
        return (index_tip < index_pip and 
                middle_tip < middle_pip and 
                ring_tip > index_pip and 
                pinky_tip > index_pip)

    def is_thumbs_up(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False

        thumb_tip = landmarks[4][2]
        thumb_ip = landmarks[3][2]

        other_tips = [landmarks[i][2] for i in [8, 12, 16, 20]]
        
        return thumb_tip < thumb_ip and all(tip > thumb_ip for tip in other_tips)

    def recognize_gesture(self, landmarks):
        for gesture_name, check_func in self.gestures.items():
            if check_func(landmarks):
                return gesture_name
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    detector = RexzeaHandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        landmarks = detector.find_position(img)

        if landmarks:
            gesture = detector.recognize_gesture(landmarks)

            cv2.putText(
                img, 
                f"Gesture: {gesture}", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

        cv2.imshow("Rexzea Hand Detector", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()