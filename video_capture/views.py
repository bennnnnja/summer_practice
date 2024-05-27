from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.shortcuts import render
from.emotion_detected import EmotionDetector
import cv2
import numpy as np

detector = EmotionDetector()  # Создаем экземпляр детектора

@csrf_exempt
def capture_image(request):
    if request.method == 'POST':
        img_data = request.POST.get('img_data')
        img_data = base64.b64decode(img_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        # Применение функции обводки лиц
        img = resize_image(img)

        if img is None:
            # Лицо не найдено, возвращаем сообщение об ошибке
            return JsonResponse({'status': 'error', 'message': 'Face not detected'})

        # Сохраняем изображение в файл
        cv2.imwrite('to_process.png', img)
        
        # Распознаем эмоцию
        emotion, probability = detector.predict_emotion(img)
        return JsonResponse({'status': 'success', 'emotion': emotion, 'probability': float(probability)})
    elif request.method == 'GET':
        # Для GET-запросов просто возвращаем пустой ответ
        return render(request, 'capture_image.html')
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

def resize_image(img): 
     
    # Преобразуем изображение в оттенки серого для улучшения производительности 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
     
    # Загружаем предобученный классификатор для обнаружения лиц 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
     
    # Ищем лица на изображении 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
     
    if len(faces) > 0: 
        # Берем первое обнаруженное лицо (можно выбрать любое, если нужно) 
        x, y, w, h = faces[0] 
         
        # Обрезаем лицо квадратом 
        face_square = img[y:y+h, x:x+w] 
         
        # Ужимаем изображение до 44x44 пикселей 
        resized_face = cv2.resize(face_square, (48, 48)) 
        gray_resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY) 
         
        return gray_resized_face 
    else: 
        print("Лицо не найдено") 
        return None 


def draw_face_borders(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image