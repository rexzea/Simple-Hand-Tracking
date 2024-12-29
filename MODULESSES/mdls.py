import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

class HandGestureRecognizer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.classifier = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()

        self.prepare_default_training()

    def prepare_default_training(self):
        training_data = [
            [0, 100, 0.5],   
            [1, 50, 0.3],    
            [2, 75, 0.4],    
            [3, 60, 0.35],   
            [4, 80, 0.45]    
        ]
        labels = [0, 1, 2, 3, 4]
        
        self.train_model(training_data, labels)

    def train_model(self, training_data, labels):
        try:
            scaled_data = self.scaler.fit_transform(training_data)

            self.classifier.fit(scaled_data, labels)
            
            print("Model berhasil dilatih!")
        
        except Exception as e:
            print(f"Kesalahan saat melatih model: {e}")

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        return thresh

    def extract_hand_features(self, thresh):
        try:
            contours, _ = cv2.findContours(
                thresh, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            hand_contour = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(hand_contour)
            perimeter = cv2.arcLength(hand_contour, True)

            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            hull = cv2.convexHull(hand_contour, returnPoints=False)
            defects = cv2.convexityDefects(hand_contour, hull)

            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    start, end, far, distance = defects[i][0]
                    if distance > 10000:
                        finger_count += 1

            features = [
                finger_count,     
                area,            
                compactness      
            ]
            
            return features
        
        except Exception as e:
            print(f"Kesalahan ekstraksi fitur: {e}")
            return None

    def recognize_gesture(self, features):
        if features is None:
            return ""
        
        try:

            scaled_features = self.scaler.transform([features])
            

            prediction = self.classifier.predict(scaled_features)
            
            return self.gestures[prediction[0]]
        
        except Exception as e:
            print(f"Kesalahan mengenali gestur: {e}")
            return ""

    def run_recognition(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            thresh = self.preprocess_frame(frame)
            
            features = self.extract_hand_features(thresh)
            
            if features:
                gesture = self.recognize_gesture(features)
                
                cv2.putText(
                    frame, 
                    gesture, 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            
            cv2.imshow("Simple Hand Gesture Recognition", frame)
            
            cv2.imshow("Threshold", thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = HandGestureRecognizer()
    
    recognizer.run_recognition()

if __name__ == "__main__":
    main()