from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializer import GeezNumberSerializer
from .models import NumbersImage
import os
import cv2 as cv
import os
from pathlib import Path
import tensorflow as tf
import numpy as np


# Get the absolute path to the current script
script_path = os.path.dirname(os.path.abspath(__file__))

# Specify the full path to cnn1.h5
model_path = os.path.join(script_path, 'model_training', 'cnn1.h5')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
classifier = tf.keras.models.load_model(model_path)

@api_view(["POST"])
def creat_numbers(request):
    data = request.data
    geez_number = NumbersImage.objects.create(
        image_url= data['image_url'],
        creation_date = data["creation_date"]                   
    )
    serializer = GeezNumberSerializer(geez_number,many=False)
    return Response(serializer.data)

@api_view(["GET"])
def predict(request):
    
    geez_numbers = NumbersImage.objects.order_by('-creation_date')
    image_path = geez_numbers[0].image_url
   
    image_url = os.path.join('numbers/',str(image_path))
    image = cv.imread(os.path.join(BASE_DIR,image_url), 0)
    number = cv.resize(image,(28,28)).reshape(784)
    word_dict = {i: str(i) for i in range(202)}       
    img_final =np.reshape(number, (-1,28,28,1))
    predicted = word_dict[np.argmax(classifier.predict(img_final))]
    
    return Response(f'{predicted}')
















         


        
        
        

    

