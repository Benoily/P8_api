#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse


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
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class Classifie(Resource):
    def post(self):
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

api.add_resource(Classifie, '/predict')


if __name__ == '__main__':
    # Load model
    model = keras.models.load_model('model/model_unet_dice_aug.h5', custom_objects={'total_loss': total_loss}, compile=True)

    app.run(debug=True)
