from flask import render_template, request
from flask import redirect,url_for
from PIL import Image
import os
from face import extract_face
from face import facenett
from face import facevideo
import cv2 


UPLOAD_FLODER = 'static/uploads'

def base():
    return render_template("base.html")

def index():
    return render_template("index.html")


def faceapp():
    return render_template("faceapp.html")

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def getname():
    if request.method == 'POST':
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        name=facenet(path,filename)     
    return name
    
def faces():
    if request.method == 'POST':
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        px = extract_face(path)
        cv2.imwrite('./static/predict/{}'.format(filename),px)
        name = facenett(path,filename)
        folder_path = (r'C:\Users\Por\Desktop\proj\data')
        test = os.listdir(folder_path)
        for images in test:
            if images.endswith(".jpg"):
                os.remove(os.path.join(folder_path, images))

        return render_template('faces.html',fileupload=True,img_name=filename, w=w,name=name)
    return render_template('faces.html',fileupload=False,img_name="freeai.png")



                    
