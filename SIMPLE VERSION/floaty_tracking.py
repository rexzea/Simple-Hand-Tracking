import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(rgb_frame)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )
        return frame

    def find_positions(self, frame, hand_number=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_number:
                hand = self.results.multi_hand_landmarks[hand_number]
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([id, cx, cy])
        return landmark_list

    def check_finger_up(self, landmark_list):
        if len(landmark_list) > 0:
            if landmark_list[8][2] < landmark_list[6][2]:
                return True
        return False

def main():
    cap = cv2.VideoCapture(0)

    tracker = HandTracker()
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frames from webcam")
            break
            
        frame = tracker.find_hands(frame)
        
        landmark_list = tracker.find_positions(frame)
        
        if tracker.check_finger_up(landmark_list):
            cv2.putText(frame, "Index Finger Detected!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Rexzea Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()