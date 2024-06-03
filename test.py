from video_capture.emotion_detected import EmotionDetector
import cv2
from video_capture.views import resize_image

detector = EmotionDetector()  # Создаем экземпляр детектора

img = cv2.imread('angry.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
emotion, probability = detector.predict_emotion(gray)
print(emotion, probability)