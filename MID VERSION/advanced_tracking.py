import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot, atan2, degrees
import json
from datetime import datetime
import collections
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    filename='hand_tracking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RexzeaMidVersionAdvancedTracking:
    def __init__(self, name: str, confidence: float, timestamp: float):
        self.name = name
        self.confidence = confidence
        self.timestamp = timestamp

class RexzeaAdvancedTracking:
    def __init__(self, 
                 mode=False, 
                 max_hands=2, 
                 detection_con=0.8, 
                 track_con=0.8,
                 gesture_smoothing=5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
            model_complexity=1  
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        self.gesture_history = collections.deque(maxlen=gesture_smoothing)

        self.hand_size_history = collections.deque(maxlen=30)
        self.is_calibrated = False
        self.calibration_frames = 0
        self.base_hand_size = None

        self.finger_angles = {}
        self.gesture_thresholds = {
            "pinch_threshold": 0.1,    
            "finger_straight_threshold": 160.0,  
            "finger_bent_threshold": 90.0,      
            "gesture_confidence_threshold": 0.85
        }
        
        self.fps_history = collections.deque(maxlen=30)
        self.last_time = time.time()
        
        self.gesture_definitions = {
            "palm": {"min_confidence": 0.85, "required_fingers": [1,1,1,1,1]},
            "fist": {"min_confidence": 0.9, "required_fingers": [0,0,0,0,0]},
            "pointing": {"min_confidence": 0.85, "required_fingers": [0,1,0,0,0]},
            "pinch": {"min_confidence": 0.8, "distance_threshold": 0.1},
            "peace": {"min_confidence": 0.85, "required_fingers": [0,1,1,0,0]},
            "ok": {"min_confidence": 0.9, "special_check": True},
            "thumbs_up": {"min_confidence": 0.85, "required_fingers": [1,0,0,0,0]},
            "thumbs_down": {"min_confidence": 0.85, "required_fingers": [1,0,0,0,0]}
        }

    def calibrate_hand_size(self, landmarks: List) -> None:
        if len(landmarks) == 21:
            wrist = landmarks[0]
            middle_finger_base = landmarks[9]
            hand_size = hypot(
                middle_finger_base[0] - wrist[0],
                middle_finger_base[1] - wrist[1]
            )
            self.hand_size_history.append(hand_size)
            
            if len(self.hand_size_history) == 30:
                self.base_hand_size = np.median(self.hand_size_history)
                self.is_calibrated = True
                logging.info(f"Hand calibration complete. Base size: {self.base_hand_size}")

    def calculate_finger_angles(self, landmarks: List) -> Dict[str, float]:
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        angles = {}

        finger_points = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger, points in finger_points.items():
            angles[finger] = []
            for i in range(len(points)-2):
                p1 = landmarks[points[i]]
                p2 = landmarks[points[i+1]]
                p3 = landmarks[points[i+2]]
                
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                angles[finger].append(angle)
        
        return angles

    def detect_gesture(self, landmarks: List) -> Tuple[str, float]:
        if not self.is_calibrated:
            self.calibrate_hand_size(landmarks)
            return "calibrating", 0.0

        self.finger_angles = self.calculate_finger_angles(landmarks)

        fingers_state = self.get_fingers_state(landmarks)

        gestures_confidence = {}
        
        for gesture_name, params in self.gesture_definitions.items():
            confidence = self.calculate_gesture_confidence(
                gesture_name, 
                fingers_state, 
                landmarks, 
                params
            )
            gestures_confidence[gesture_name] = confidence
        
        best_gesture = max(gestures_confidence.items(), key=lambda x: x[1])

        if best_gesture[1] >= self.gesture_thresholds["gesture_confidence_threshold"]:
            self.gesture_history.append(best_gesture[0])

        if self.gesture_history:
            smooth_gesture = collections.Counter(self.gesture_history).most_common(1)[0]
            return smooth_gesture[0], best_gesture[1]
        
        return "unknown", 0.0

    def calculate_gesture_confidence(self, 
                                  gesture_name: str, 
                                  fingers_state: List[int], 
                                  landmarks: List,
                                  params: Dict) -> float:
        if gesture_name == "pinch":
            return self.calculate_pinch_confidence(landmarks)
        elif gesture_name == "ok":
            return self.calculate_ok_sign_confidence(landmarks)
        else:
            return self.calculate_standard_gesture_confidence(
                fingers_state, 
                params.get("required_fingers", []),
                params.get("min_confidence", 0.0)
            )

    def calculate_pinch_confidence(self, landmarks: List) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        normalized_distance = distance / self.base_hand_size if self.base_hand_size else float('inf')
        
        if normalized_distance < self.gesture_thresholds["pinch_threshold"]:
            confidence = 1.0 - (normalized_distance / self.gesture_thresholds["pinch_threshold"])
            return min(1.0, max(0.0, confidence))
        return 0.0

    def calculate_ok_sign_confidence(self, landmarks: List) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        circle_formation = hypot(thumb_tip[0] - index_tip[0], 
                               thumb_tip[1] - index_tip[1])
        normalized_circle = circle_formation / self.base_hand_size if self.base_hand_size else float('inf')
        
        if 0.1 <= normalized_circle <= 0.3:
            return 1.0 - abs(0.2 - normalized_circle) / 0.2
        return 0.0

    def calculate_standard_gesture_confidence(self, 
                                           current_state: List[int], 
                                           required_state: List[int],
                                           min_confidence: float) -> float:
        if len(current_state) != len(required_state):
            return 0.0
            
        matches = sum(1 for a, b in zip(current_state, required_state) if a == b)
        confidence = matches / len(required_state)
        
        return confidence if confidence >= min_confidence else 0.0

    def get_fingers_state(self, landmarks: List) -> List[int]:
        fingers = []

        thumb_angle = sum(self.finger_angles['thumb'])
        fingers.append(1 if thumb_angle > self.gesture_thresholds["finger_straight_threshold"] else 0)

        for finger in ['index', 'middle', 'ring', 'pinky']:
            if sum(self.finger_angles[finger]) > self.gesture_thresholds["finger_straight_threshold"]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        detection_results = {
            "hands_detected": 0,
            "gestures": [],
            "fps": 0,
            "calibration_status": "calibrated" if self.is_calibrated else "calibrating"
        }
        
        if results.multi_hand_landmarks:
            detection_results["hands_detected"] = len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    landmarks.append([int(lm.x * w), int(lm.y * h)])

                gesture_name, confidence = self.detect_gesture(landmarks)

                detection_results["gestures"].append({
                    "hand_index": hand_idx,
                    "gesture": gesture_name,
                    "confidence": confidence,
                    "handedness": results.multi_handedness[hand_idx].classification[0].label
                })

                self.draw_hand_landmarks(frame, hand_landmarks, gesture_name, confidence)

        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_history.append(fps)
        detection_results["fps"] = int(np.mean(self.fps_history))

        self.draw_info(frame, detection_results)
        
        return frame, detection_results

    def draw_hand_landmarks(self, 
                          frame: np.ndarray, 
                          landmarks, 
                          gesture: str, 
                          confidence: float) -> None:
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmarks_style(),
            self.drawing_styles.get_default_hand_connections_style()
        )
        
        h, w, _ = frame.shape
        landmark_point = landmarks.landmark[0]
        position = (int(landmark_point.x * w), int(landmark_point.y * h))
        
        cv2.putText(frame, 
                   f"{gesture} ({confidence:.2f})", 
                   (position[0] - 10, position[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (0, 255, 0), 
                   2)

    def draw_info(self, frame: np.ndarray, results: Dict) -> None:
        y_pos = 30
        line_height = 30
        
        cv2.putText(frame,
                   f"FPS: {results['fps']}",
                   (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0),
                   2)
        y_pos += line_height

        cv2.putText(frame,
                   f"Status: {results['calibration_status']}",
                   (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0) if results['calibration_status'] == "calibrated" else (0, 165, 255),
                   2)
        y_pos += line_height

        cv2.putText(frame,
                   f"Hands Detected: {results['hands_detected']}",
                   (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0),
                   2)

        if not self.is_calibrated:
            h, w, _ = frame.shape
            cv2.rectangle(frame,
                         (int(w*0.25), int(h*0.25)),
                         (int(w*0.75), int(h*0.75)),
                         (0, 165, 255),
                         2)
            cv2.putText(frame,
                       "Place hand in box for calibration",
                       (int(w*0.25), int(h*0.23)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 165, 255),
                       2)

    def save_detection_data(self, results: Dict) -> None:
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        try:
            with open("hand_tracking_log.json", "a") as f:
                json.dump(data, f)
                f.write("\n")
        except Exception as e:
            logging.error(f"Error saving detection data: {e}")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = RexzeaAdvancedTracking(
        detection_con=0.8,
        track_con=0.8,
        gesture_smoothing=5
    )

    cv2.namedWindow("Rexzea Hand Tracking", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logging.error("Failed to read frame from camera")
                break

            processed_frame, results = tracker.process_frame(frame)

            tracker.save_detection_data(results)

            cv2.imshow("Rexzea Advanced Hand Tracking", processed_frame)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'): 
                tracker.is_calibrated = False
                tracker.hand_size_history.clear()
                logging.info("Calibration reset")
    
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical error: {e}")
