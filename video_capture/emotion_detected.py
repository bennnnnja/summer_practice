import tensorflow as tf
import numpy as np

class EmotionDetector:
    def __init__(self, model_path='model120.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    def predict_emotion(self, image):
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        predicted_index = np.argmax(prediction)
        return self.emotions[predicted_index], prediction[0][predicted_index]
