import numpy as np 
import pandas as pd 
import cv2 
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from PIL import Image
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import glob


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]

def facenett(path,filename):
    image = Image.open(path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize((160,160))
        face_array = np.asarray(image)
        #cv2.imwrite('./static/predict/{}.jpg'.format(filename),face_array)
        cv2.imwrite('./data/img_{}.jpg'.format(i),face_array)
    data = list()
    for i in glob.glob('data/*.jpg', recursive=True):
        data.append(cv2.imread(i))
    emdTestX = list()
    for face in data:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)
    emdTestX = np.asarray(emdTestX)
    emdTestX_norm = in_encoder.transform(emdTestX)
    yhat_test = model.predict(emdTestX_norm)
    print(yhat_test)
    predict_name = out_encoder.inverse_transform(yhat_test)
    print(predict_name)
    print(list(out_encoder.classes_))
    
    for i in range(len(yhat_test)):
        print(predict_name[i])
        predict_name = out_encoder.inverse_transform(yhat_test)
        df.at[yhat_test[i],'score'] = 1
    writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    return predict_name

def facevideo():
    data = list()
    for i in glob.glob('data/*.jpg', recursive=True):
        data.append(cv2.imread(i))
    emdTestX = list()
    for face in data:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)
    emdTestX = np.asarray(emdTestX)
    emdTestX_norm = in_encoder.transform(emdTestX)
    yhat_test = model.predict(emdTestX_norm)
    print(yhat_test)
    predict_name = out_encoder.inverse_transform(yhat_test)
    print(predict_name)
    for i in range(len(yhat_test)):
        print(yhat_test[i])
        predict_name = out_encoder.inverse_transform(yhat_test)
        df.at[yhat_test[i],'score'] = 1
    writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    

data = np.load('faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
emdd = np.load('faces-embeddings.npz')
emdTrainX, trainy, emdTestX, testy = emdd['arr_0'], emdd['arr_1'], emdd['arr_2'], emdd['arr_3']
facenet_model = load_model('./facenet_keras.h5')
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)
model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)
df = pd.read_csv('score.csv')



