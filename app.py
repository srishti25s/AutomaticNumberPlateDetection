from flask import Flask, render_template, request, redirect,url_for
from testing import start_video
from os import listdir
from os.path import isfile, join


app = Flask(__name__)
//rtsp://100.91.39.237:5554

VIDEO_UPLOAD_FILE_PATH = "static/videos"
IMAGE_PATH = "static/plate_result"

@app.route("/")
def home():
        files_list = [f for f in listdir(VIDEO_UPLOAD_FILE_PATH) if isfile(join(VIDEO_UPLOAD_FILE_PATH, f))]
        return render_template("home.html",files=files_list)

@app.route("/renderVideo",methods=["POST"])
def render_video():
    start_video(str(request.data))
    return "Success"
    
@app.route("/numberPlatesList",methods=["POST"])
def render_number_plate():
    files_list = [f for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))]
    return ','.join(files_list)

if __name__ == "__main__":
    app.run(debug=True)