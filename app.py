from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import cv2 as cv
import io
from werkzeug.utils import secure_filename,send_from_directory

app = Flask(__name__)

Allowed_Image_Extenxion = ['jpg','jpeg','png']
Allowed_Video_Extension = ['mp4','avi']

classes = []
with open('classes.txt','r') as f:
    for line in f:
        classes.append(line.strip())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def img_detect():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            f.save(filepath)
            
            global img_path 
            img_detect.img_path = filepath
            
            file_extension = f.filename.rsplit('.',1)[1].lower()
            
            if file_extension in Allowed_Image_Extenxion:
                img = cv.imread(filepath)
                frame = cv.imencode('.jpg', cv.UMat(img))[1].tobytes()
                
                image = Image.open(io.BytesIO(frame))
                
                yolo = YOLO('bestNew.pt')
                
                # result = yolo.predict(image)[0]

                project_path = os.path.join("C:", "Users", "Hasan", "Desktop", "Final API", "Final 3 API's", "Gender API NEW", "runs", "detect")
                detection = yolo.predict(image, save=True, project=project_path)
                # yolo.predict(image, save=True)

                # detection = yolo.predict(image, save=True, imgsz=640, conf=0.5, project="C:\Users\Hasan\Desktop\Final API\Final 3 API's\Gender API NEW\runs\detect")
                
                return display(f.filename)
            
                
                # #finding the bounding box, confidence score and class id from the result
                # boxes = result.boxes.xyxy.tolist()
                # confidences = result.boxes.conf.tolist()
                # class_ids = result.boxes.cls.tolist()
                
                # #Finding the name of the classes of the corresponding class id
                # # class_names = [yolo.names[class_id] if class_id < len(yolo.names) else "Unknown Class" for class_id in class_ids] 
                
                # #Draw bounding box, confidence score and lables on the image
                # detected_image = cv.cvtColor(result.orig_img, cv.COLOR_RGB2BGR)
                
                # for i in range(len(boxes)):
                #     box = boxes[i]
                #     confidence = confidences[i]
                #     # label = class_names[i]
                #     label = classes[i] if i < len(classes) else "Unknown"

                    
                #     #Draw bounding box
                #     x, y, w, h = map(int,box)
                #     color = (255, 255, 0) #Drawing color line for the box
                #     cv.rectangle(detected_image,(x,y),(w,h),color,2)
                    
                #     #Add label and Confidence score
                #     label_text = f'{label} : {confidence:.2f}'
                #     cv.putText(detected_image,label_text,(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.9,color,2) 
                # #Save the ditected Image
                # detected_image_path = os.path.join(basepath,'Detected_Images',f.filename)
                # cv.imwrite(detected_image_path,detected_image)
                
                # #prepare the json file
                # response_data = {
                #     # 'Labels' : class_names,
                #     'Labels': classes,
                #     'Confidence Score' :  confidences,
                #     'Boxes' : boxes
                # }
                
                # return jsonify(response_data)
                
            elif file_extension in Allowed_Video_Extension:
                video_path = filepath
                cap = cv.VideoCapture(video_path)
                
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter('output.mp4',fourcc, 20.0, (frame_width,frame_height))
                
                model = YOLO('bestNew.pt')
                
                
                
                return 'Video'
            
    return "False File Format"
                
                
@app.route('/<path:filename>')                
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,f))]
    
    if not subfolders:
        return "No detection results found."
    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path,x)))
    directory = folder_path + '/' + latest_subfolder
    files = os.listdir(directory)
    
    if not files:
        return "No files found in the detection results folder."
    
    latest_file = files[0]
    
    filename = os.path.join(folder_path,latest_subfolder,latest_file)  
    
    file_extension = filename.rsplit('.',1)[1].lower()
    
    environ = request.environ
    if file_extension in Allowed_Image_Extenxion:
        return send_from_directory(directory,latest_file,environ)
    else:
        return "Invalid FIle Format"

if __name__ == '__main__':
    app.run(debug=True)
