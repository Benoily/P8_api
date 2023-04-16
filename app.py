"""http://web.univ-ubs.fr/lmba/lardjane/python/c4.pdf -> page 260"""

from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import os
from flask import Flask,request,url_for, jsonify
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


app = Flask(__name__)
PORT = os.environ.get("PORT")

# Chargement du modèle
# Load model
# Load model
model = keras.models.load_model('model/model_unet_dice_aug.h5', custom_objects={'total_loss': total_loss}, compile=True)


# defining a route for only post requests
@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    try:
        r = request
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # build a response dict to send back to client
        response = {'message': 'image received. size={}x{}'.format(
            img.shape[1], img.shape[0])}
        
        img = img/255
        x = cv2.resize(img, (256, 256))
        pred = model.predict(np.expand_dims(x, axis=0))
        pred_mask = np.argmax(pred, axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_mask = np.squeeze(pred_mask)

        # creating a response object
        # storing the model's prediction in the object
        response['prediction'] =  pred_mask.tolist()
    except Exception as e:
        response['error'] = e

    # returning the response object as json
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=PORT)
#     from waitress import serve
#     serve(app, host='0.0.0.0')
