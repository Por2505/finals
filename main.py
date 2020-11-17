import views
from flask import Flask, Response
from flask import render_template, request
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import time
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)

app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\\Users\\Por\\Desktop\\proj\\database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/faceapp','faceapp',views.faceapp)
app.add_url_rule('/faces','faces',views.faces,methods=['GET','POST'])


@app.route('/detect')
def detect():
    return render_template('detect.html')
def detect():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    img_id = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            frame = cv2.VideoCapture(0)
            continue
        
        if ret:
            frame = np.asarray(frame)
            try:
                results = detector.detect_faces(frame)
                for i in range(len(results)):
                    x1, y1, width, height = results[i]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    pixels = np.asarray(frame)
                    face = pixels[y1:y2, x1:x2]
                    image = Image.fromarray(face)
                    image = image.resize((160,160))
                    face_array = np.asarray(image)
                    cv2.imwrite('./data/img_{}.jpg'.format(i),face_array)
                    facevideo()
                
               
                
            except:
                print("Something else went wrong")
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.1)
        #key = cv2.waitKey(20)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
@app.route('/detect_feed')
def detect_feed():
    global video
    return Response(detect(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])




@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)