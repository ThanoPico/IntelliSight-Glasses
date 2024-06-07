import cv2
import numpy as np
import tensorflow as tf
import pygame



model = tf.keras.models.load_model('teachable_machine.h5')

class_names = ['Tree', 'Wall', 'Human', ...]

pygame.mixer.init()


while True:

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:

            break


        resized_frame = cv2.resize(frame, (224, 224))

        expanded_frame = np.expand_dims(resized_frame, axis=0)

    

        cv2.imshow('Frame', frame)


        predictions = model.predict(expanded_frame)

        predicted_class = np.argmax(predictions)

        class_name = class_names[predicted_class]


        if class_name == 'Tree':

            pygame.mixer.music.load('Tree.mp3')

            pygame.mixer.music.play()

        elif class_name == 'Wall':

            pygame.mixer.music.load('Wall.mp3')

            pygame.mixer.music.play()

        elif class_name == 'Human':

            pygame.mixer.music.load('Human.mp3')

            pygame.mixer.music.play()

    
     

    cap.release()

    cv2.destroyAllWindows()