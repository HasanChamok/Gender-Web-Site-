from flask import Flask, request, jsonify,render_template, Response, send_file, url_for
import os
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
from werkzeug.utils import secure_filename,send_from_directory
import time
import datetime
import requests
import re
from re import DEBUG,sub


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

Allowed_Image_Extenxion = ['jpg','jpeg','png']
Allowed_Video_Extension = ['mp4','avi']

classes = []
result = {}
with open('classes.txt','r') as f:
    for line in f:
        classes.append(line.strip())

@app.route("/",methods=["GET","POST"])
def main():
    #Taking the input of the files
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            fileExtension = f.filename.rsplit('.',1)[1].lower()
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            f.save(filepath)
            
            if fileExtension in Allowed_Image_Extenxion:
                result = predict_image(f,filepath)
                
            elif fileExtension in Allowed_Video_Extension:    
                result = predict_video(f,filepath)
            
    return result

def predict_image(f,filepath):
    
    basepath = os.path.dirname(__file__)
    #loading the model
    yolo = YOLO('bestNew.pt')
    #Opening the Image file
    file = Image.open(filepath)
    #putting the result file in the result 
    # result = yolo.predict(file)[0]
    result = yolo.predict(file, save=True, conf=0.4, iou=0.7)
    
    return display(f.filename)
    # return jsonify({"File Path":"Successfull"})

def predict_video(f,filepath):
    baesname = os.path.dirname(__file__) #Getting the basename of the folder
    
    cap = cv.VideoCapture(filepath) #capturing the video from the video filepath

    # Get video dimensions 
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    #Define codec and video writer Object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 30, (frame_width, frame_height))
    # out = cv.VideoWriter(os.path.join(baesname, 'Detected_Videos', 'output.mp4'), fourcc, 30, (frame_width, frame_height))
    yolo = YOLO('bestNew.pt') #Load the yolo model

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        #Do yolo detection on the frame here
        results = yolo.predict(frame, save=True)
        cv.waitKey(1)

        res_plotted = results[0].plot()
        cv.imshow('result', res_plotted)

        # Write the frame to the output video
        out.write(res_plotted)

        if cv.waitKey(1) == ord('q'):
            break

        return video_feed()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,f))]
    
    # if not subfolders:
    #     return "No detection results found."
    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path,x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename
    return render_template('index.html',image_path=image_path)

@app.route('/<path:filename>')                
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,f))]
    
    # if not subfolders:
    #     return "No detection results found."
    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path,x)))
    directory = folder_path + '/' + latest_subfolder
    files = os.listdir(directory)
    
    # if not files:
    #     return "No files found in the detection results folder."
    
    latest_file = files[0]
    
    filename = os.path.join(folder_path,latest_subfolder,latest_file)  
    
    file_extension = filename.rsplit('.',1)[1].lower()
    
    environ = request.environ
    if file_extension in Allowed_Image_Extenxion:
        return send_from_directory(directory,latest_file,environ)
    else:
        return "Invalid FIle Format"


def get_frame():
    folder_path = os.getcwd()
    mp4files = 'output.mp4'
    video = cv.VideoCapture(mp4files)
    while True:
        success,frame = video.read()
        if not success:
            break
        ret, jpeg = cv.imencode('.jpg',frame)
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)