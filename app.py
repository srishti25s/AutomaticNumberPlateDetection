from flask import Flask, render_template, request, redirect,url_for,Response
from testing import start_video
import os

import shutil
import cv2

app = Flask(__name__)

VIDEO_UPLOAD_FILE_PATH = "static/videos"
IMAGE_PATH = "static/plate_result"

if os.path.exists(IMAGE_PATH):
	shutil.rmtree(IMAGE_PATH)

os.mkdir(IMAGE_PATH)

@app.route("/")
def home():
    files_list = [f for f in os.listdir(VIDEO_UPLOAD_FILE_PATH) if os.path.isfile(os.path.join(VIDEO_UPLOAD_FILE_PATH, f))]
    return render_template("home.html",files=files_list)

@app.route("/renderVideo",methods=["POST"])
def render_video():
    path = request.data.decode("utf-8") 
    print('Path -',path)
    start_video(path)
    return "Success"

@app.route("/numberPlatesList",methods=["POST"])
def render_number_plate():
    files_list = [f for f in os.listdir(IMAGE_PATH) if os.path.isfile(os.path.join(IMAGE_PATH, f))]
    return ','.join(files_list)

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)
