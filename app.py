from flask import Flask,request,render_template,url_for
import tensorflow
from tensorflow import keras
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import base64
from io import BytesIO
#import pickle
import numpy as np
import json

#IMG_WIDTH_HEIGHT = (256, 256)
img_size = (256, 256)

# Fonctions loss
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.cast(K.flatten(y_true), K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

# Load model
model = keras.models.load_model('model/model_unet_dice_aug.h5', custom_objects={'total_loss': total_loss}, compile=True)

app =Flask(__name__)


@app.route('/')
def index():
    image_list = os.listdir('data/test/images')
    return render_template('index.html', image_list=image_list)

#@app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method =='POST':
        # Retourner le fichier sélectionné
        image = request.form['image']
        image_path = str('data/test/images/' + image)
        
        with open(image_path, "rb") as f:
            original_img_b64 = base64.b64encode(f.read())
            original_img_b64_str = original_img_b64.decode("utf-8")
        
            input_img = Image.open(
            BytesIO(base64.b64decode(original_img_b64))
        ).resize(img_size)
        
            pred_mask = np.squeeze(
                    np.argmax(
                        model.predict(np.expand_dims(input_img, 0)), axis=-1
                    )
                )
        #image=os.path.join(app.config['UPLOAD_FOLDER'],image)
        #X = img_to_array(load_img(image_path, target_size=(IMG_WIDTH_HEIGHT)))/255
        #X = np.expand_dims(X, 0)
        #prediction = model.predict(X)
        #pred_mask = np.argmax(prediction, axis=-1)
        #pred_mask = np.squeeze(pred_mask)
        print(pred_mask)
        
    return render_template('index.html',prediction=pred_mask)


if __name__ == '__main__':
    app.run(debug=True)
