from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
from .emotion_detected import EmotionDetector

detector = EmotionDetector()  # Создаем экземпляр детектора

@csrf_exempt
def capture_image(request):
    if request.method == 'POST':
        img_data = request.POST.get('img_data')
        img_data = base64.b64decode(img_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        img = resize_image(img)

        if img is None:
            return JsonResponse({'status': 'error', 'message': 'Face not detected'})

        cv2.imwrite('to_process.png', img)
        
        emotion, probability = detector.predict_emotion(img)
        return JsonResponse({'status': 'success', 'emotion': emotion, 'probability': float(probability)})
    elif request.method == 'GET':
        return render(request, 'capture_image.html')
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)


def resize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_square = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_square, (48, 48))
        gray_resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        return gray_resized_face
    else:
        return None
