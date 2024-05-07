import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
import keras
import tensorflow_hub as hub
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image, ImageOps,ImageTk

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gecmdvent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'cancer_model.h5'

# Load your trained model
model = keras.models.load_model(MODEL_PATH,custom_objects={'KerasLayer':hub.KerasLayer})

print('Model loaded. Start serving...')

classes = {
            0:'Benign',
            1:'Malignant_Pre_B',
            2:'Malignant_Pro_B',
            3:'Malignant_early_Pre_B',
           }

def model_predict(img_path, model):
    global predictions
    global final_prediction
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions[0])
    final_prediction = classes[np.argmax(score)]
    return final_prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    print("inside about")
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("inside post")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        percentage_list = []
        percentage_list.append(final_prediction)
        for i in range(len(predictions[0])):
            percentage_list.append("{0:.2%}".format(predictions[0][i]))
        print(percentage_list)
        return percentage_list
    return None


if __name__ == '__main__':
    app.run(debug=True)
