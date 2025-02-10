import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller, KeyCode
import time
import win32gui
import win32con
import win32api

class RexzeaHandController:
    def __init__(self):
        # ins MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # style landmark
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(
            color=(255, 255, 255),
            thickness=2
        )
        
        # Iins keyboard controller
        self.keyboard = Controller()
        
        # Parameter movement
        self.prev_hand_center = None
        self.movement_threshold = 50
        self.cooldown = 0.2
        self.last_action_time = 0
        
        # Parameter window
        self.WINDOW_NAME = 'Subway Surfers Hand Controller'
        
        # Parameter clench
        self.fist_cooldown = 0.5
        self.last_fist_time = 0
        self.is_skateboard_active = False
        
        # UI elements
        self.overlay_alpha = 0.3
        self.font = cv2.FONT_HERSHEY_DUPLEX
        
    def create_status_bar(self, frame, status_height=80):
        h, w = frame.shape[:2]
        status_bar = np.zeros((status_height, w, 3), dtype=np.uint8)
        overlay = frame.copy()
        overlay[:status_height] = status_bar
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        return frame
    
    def draw_control_guide(self, frame):
        h, w = frame.shape[:2]
        guide_height = 120
        
        # guide area
        guide_bar = np.zeros((guide_height, w, 3), dtype=np.uint8)
        overlay = frame.copy()
        overlay[h-guide_height:] = guide_bar
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        
        # guide
        base_y = h - guide_height + 30
        guides = [
            "CONTROLS GUIDE:",
            "→ Move Hand Left/Right: Move Left/Right",
            "→ Move Hand Up/Down: Jump/Roll",
            "→ Make Fist: Activate Skateboard",
            "→ Press 'Q': Quit"
        ]
        
        for i, text in enumerate(guides):
            cv2.putText(frame, text, (20, base_y + i*20),
                       self.font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def draw_status_info(self, frame, action=None):
        # skateboard status
        skateboard_status = "ACTIVE" if self.is_skateboard_active else "INACTIVE"
        status_color = (0, 255, 0) if self.is_skateboard_active else (0, 0, 255)
        
        # draw skateboard
        cv2.putText(frame, f"SKATEBOARD: {skateboard_status}", 
                   (20, 30), self.font, 0.8, status_color, 2)
        
        # current action
        if action:
            action_text = f"ACTION: {action}"
            cv2.putText(frame, action_text, (20, 60),
                       self.font, 0.8, (255, 165, 0), 2)
        
        return frame
    
    def draw_hand_tracking_info(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_drawing_spec,
            self.connection_drawing_spec
        )
        return frame

    def detect_fist(self, hand_landmarks, frame_shape):
        h, w = frame_shape[:2]
        finger_tips = [8, 12, 16, 20]
        finger_mids = [6, 10, 14, 18]
        
        fingers_bent = 0
        for tip, mid in zip(finger_tips, finger_mids):
            tip_y = hand_landmarks.landmark[tip].y * h
            mid_y = hand_landmarks.landmark[mid].y * h
            if tip_y > mid_y:
                fingers_bent += 1
        
        thumb_tip = hand_landmarks.landmark[4].x * w
        thumb_base = hand_landmarks.landmark[2].x * w
        thumb_bent = thumb_tip < thumb_base
        
        return fingers_bent >= 3 and thumb_bent
    
    def toggle_skateboard(self, is_fist):
        current_time = time.time()
        if is_fist and not self.is_skateboard_active and current_time - self.last_fist_time > self.fist_cooldown:
            self.keyboard.press(KeyCode.from_char(' '))
            self.is_skateboard_active = True
            self.last_fist_time = current_time
        elif not is_fist and self.is_skateboard_active:
            self.keyboard.release(KeyCode.from_char(' '))
            self.is_skateboard_active = False
    
    def process_frame(self, frame):
        # horizontal mirror frame
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        action = None
        
        # bar status
        frame = self.create_status_bar(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # landmark
            frame = self.draw_hand_tracking_info(frame, hand_landmarks)
            
            # clench detection
            is_fist = self.detect_fist(hand_landmarks, frame.shape)
            self.toggle_skateboard(is_fist)
            
            # movement detection
            hand_center = self.calculate_hand_center(hand_landmarks, frame)
            if time.time() - self.last_action_time > self.cooldown:
                action = self.detect_movement(hand_center)
                if action:
                    self.last_action_time = time.time()
            
            self.prev_hand_center = hand_center
        
        # status
        frame = self.draw_status_info(frame, action)
        
        # control guide
        frame = self.draw_control_guide(frame)
        
        return frame, action
    
    def calculate_hand_center(self, hand_landmarks, frame):
        h, w, _ = frame.shape
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x * w)
            y_coords.append(landmark.y * h)
        
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def detect_movement(self, current_center):
        if self.prev_hand_center is None:
            return None
            
        dx = current_center[0] - self.prev_hand_center[0]
        dy = current_center[1] - self.prev_hand_center[1]
        
        if abs(dx) > self.movement_threshold:
            if dx > 0:
                self.send_keyboard_input(Key.right)
                return "RIGHT"
            else:
                self.send_keyboard_input(Key.left)
                return "LEFT"
        elif abs(dy) > self.movement_threshold:
            if dy > 0:
                self.send_keyboard_input(Key.down)
                return "DOWN"
            else:
                self.send_keyboard_input(Key.up)
                return "UP"
        
        return None
    
    def send_keyboard_input(self, key):
        self.keyboard.press(key)
        time.sleep(0.1)
        self.keyboard.release(key)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        # camera retolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        window_setup_delay = 10
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, action = self.process_frame(frame)
            cv2.imshow(self.WINDOW_NAME, processed_frame)
            
            frame_count += 1
            if frame_count == window_setup_delay:
                self.make_window_always_on_top()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.keyboard.release(KeyCode.from_char(' '))
        cap.release()
        cv2.destroyAllWindows()
    
    def make_window_always_on_top(self):
        hwnd = win32gui.FindWindow(None, self.WINDOW_NAME)
        if hwnd:
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            window_width = screen_width // 3
            window_height = (window_width * 3) // 4
            x_pos = screen_width - window_width
            y_pos = 0
            
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,
                                x_pos, y_pos, window_width, window_height,
                                win32con.SWP_SHOWWINDOW)
            
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            style |= win32con.WS_EX_TOPMOST
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

if __name__ == "__main__":
    controller = RexzeaHandController()
    controller.run()