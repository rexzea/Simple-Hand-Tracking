import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot
import pygame
import random
from datetime import datetime

class RexzeaGameTracking:
    def __init__(self, mode=False, max_hands=2, detection_con=0.8, track_con=0.8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # pygame 
        pygame.init()
        pygame.mixer.init()
        
        # sound effects
        try:
            self.punch_sound = pygame.mixer.Sound('punch.wav')
            self.whoosh_sound = pygame.mixer.Sound('whoosh.wav')
        except:
            print("Sound files not found. Continuing without sound effects.")
        
        self.score = 0
        self.particles = []
        self.virtual_objects = []
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.combo_count = 0
        
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255)
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.drawing_points = []
        self.current_drawing = []

        self.game_mode = "normal"  
        
    def detect_gesture(self, hand_landmarks, frame_shape):
        h, w, _ = frame_shape
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.append([int(lm.x * w), int(lm.y * h)])

        fingers = self.count_fingers(landmarks)

        if sum(fingers) == 0: 
            return "fist"
        elif sum(fingers) == 5:  
            return "palm"
        elif fingers == [0, 1, 0, 0, 0]: 
            return "pointing"
        elif fingers == [0, 1, 1, 0, 0]: 
            return "peace"
        elif self.check_ok_gesture(landmarks):  
            return "ok"
        
        return "unknown"

    def count_fingers(self, landmarks):
        fingers = []
        
        tip_ids = [4, 8, 12, 16, 20]  
        base_ids = [2, 6, 10, 14, 18]  

        if landmarks[tip_ids[0]][0] < landmarks[base_ids[0]][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if landmarks[tip_ids[id]][1] < landmarks[base_ids[id]][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def check_ok_gesture(self, landmarks):
        distance = hypot(
            landmarks[4][0] - landmarks[8][0],
            landmarks[4][1] - landmarks[8][1]
        )
        return distance < 20

    def add_particle_effect(self, position, gesture):
        if gesture == "fist":
            for _ in range(20):
                angle = random.uniform(0, 2 * np.pi)
                speed = random.uniform(5, 15)
                self.particles.append({
                    'pos': list(position),
                    'vel': [speed * np.cos(angle), speed * np.sin(angle)],
                    'life': 20,
                    'color': self.colors['red']
                })
        elif gesture == "palm":
            for _ in range(15):
                angle = random.uniform(0, 2 * np.pi)
                speed = random.uniform(3, 10)
                self.particles.append({
                    'pos': list(position),
                    'vel': [speed * np.cos(angle), speed * np.sin(angle)],
                    'life': 15,
                    'color': self.colors['yellow']
                })

    def update_particles(self):
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def draw_particles(self, frame):
        for particle in self.particles:
            pos = tuple(map(int, particle['pos']))
            cv2.circle(frame, pos, 2, particle['color'], -1)

    def process_gesture_action(self, frame, gesture, hand_position):
        h, w, _ = frame.shape
        
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_cooldown = time.time()
            
            if gesture == "fist":
                # efek pukulan
                self.add_particle_effect(hand_position, "fist")
                self.score += 10
                self.combo_count += 1
                try:
                    self.punch_sound.play()
                except:
                    pass
                
            elif gesture == "palm":
                # efek partikel 
                self.add_particle_effect(hand_position, "palm")
                self.score += 5
                try:
                    self.whoosh_sound.play()
                except:
                    pass
                
            elif gesture == "pointing":
                if self.game_mode != "drawing":
                    self.game_mode = "drawing"
                    self.current_drawing = []
                
            elif gesture == "peace":
                # mereset score
                self.score = 0
                self.combo_count = 0

    def draw_game_interface(self, frame, gesture):
        h, w, _ = frame.shape

        cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                    self.font, 1, self.colors['green'], 2)
        if self.combo_count > 1:
            cv2.putText(frame, f"Combo: x{self.combo_count}", (10, 70), 
                       self.font, 1, self.colors['yellow'], 2)

        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 20), 
                    self.font, 1, self.colors['white'], 2)

        cv2.putText(frame, f"Mode: {self.game_mode}", (w - 200, 30), 
                    self.font, 1, self.colors['blue'], 2)

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                gesture = self.detect_gesture(hand_landmarks, frame.shape)

                hand_center = tuple(map(int, [
                    hand_landmarks.landmark[9].x * frame.shape[1],
                    hand_landmarks.landmark[9].y * frame.shape[0]
                ]))

                self.process_gesture_action(frame, gesture, hand_center)

                self.update_particles()
                self.draw_particles(frame)

                self.draw_game_interface(frame, gesture)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = RexzeaGameTracking()

    cv2.namedWindow("Interactive Hand Tracking", cv2.WINDOW_NORMAL)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        processed_frame = tracker.process_frame(frame)

        cv2.imshow("Rexzea Mini Game Hand Tracking", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
