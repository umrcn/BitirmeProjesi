import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('SkinCancer.h5')
model_brain =load_model('BrainTumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Malign melanom tespit edilmedi."
	elif classNo==1:
		return "Maalesef Malign melanom tespit edildi."


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=np.argmax(model.predict(input_img), axis=-1)
    return result

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)

        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None





#brain Teumor code
def get_className_brain(classNo):
	if classNo==0:
		return "Beyin tümörü tespit edilmedi."
	elif classNo==1:
		return "Tümor tespit edildi."

def getResult_brain(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=np.argmax(model_brain.predict(input_img), axis=-1)
    print(result)
    return result


@app.route('/index_brain', methods=['GET'])
def index_brain():
    return render_template('index_brain.html')


@app.route('/predict_brain', methods=['GET', 'POST'])
def upload_brain():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        value=getResult_brain(file_path)
        result=get_className_brain(value) 
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)