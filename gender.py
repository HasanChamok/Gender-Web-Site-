from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
from PIL import Image
import cv2 as cv

app = Flask(__name__)

Allowed_Image_Extenxion = ['jpg','jpeg','png']
Allowed_Video_Extension = ['mp4','avi']

classes = []
result = {}
with open('classes.txt','r') as f:
    for line in f:
        classes.append(line.strip())

def predict_image(f,filepath):
    
    basepath = os.path.dirname(__file__)
    #loading the model
    yolo = YOLO('bestNew.pt')
    #Opening the Image file
    file = Image.open(filepath)
    #putting the result file in the result 
    result = yolo.predict(file)[0]
    
    #finding the bounding box, confidence score and class id from the result
    boxes = result.boxes.xyxy.tolist()
    confidences = result.boxes.conf.tolist()
    class_ids = result.boxes.cls.tolist()
    
    #Finding the name of the classes of the corresponding class id
    # class_names = [yolo.names[class_id] if class_id < len(yolo.names) else "Unknown Class" for class_id in class_ids] 
    
    #Draw bounding box, confidence score and lables on the image
    detected_image = cv.cvtColor(result.orig_img, cv.COLOR_RGB2BGR)

    response_data = []
    
    for i in range(len(boxes)):
        box = boxes[i]
        confidence = confidences[i]
        # label = class_names[i]
        label = classes[i] if i < len(classes) else "Unknown"

        
        #Draw bounding box
        x, y, w, h = map(int,box)
        color = (255, 255, 0) #Drawing color line for the box
        cv.rectangle(detected_image,(x,y),(w,h),color,2)
        
        #Add label and Confidence score
        label_text = f'{label} : {confidence:.2f}'
        cv.putText(detected_image,label_text,(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.9,color,2) 

        #prepare the json file
        response_data.append ({
            # 'Labels' : class_names,
            'Labels': label,
            'Confidence Score' :  confidence,
            'Boxes' : box
        })
    #Save the ditected Image
    detected_image_path = os.path.join(basepath,'Detected_Images',f.filename)
    cv.imwrite(detected_image_path,detected_image)
    
    
    
    return jsonify(response_data)
    # return jsonify({"File Path":filepath})

def predict_video(f,filepath):
    baesname = os.path.dirname(__file__) #Getting the basename of the folder
    
    yolo = YOLO('bestNew.pt') #Load the yolo model
    cap = cv.VideoCapture(filepath) #capturing the video from the video filepath

    output_data = [] #For saving the output result
    
    fps = cap.get(cv.CAP_PROP_FPS) #Getting the FPS of the video
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) #Getting the frame heght
    frmae_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) #Getting the frame width
    
    # out = cv.VideoWriter(os.path.join(baesname,'Detected_Videos',f.filename),cv.VideoWriter.fourcc('M','J','P','G'),fps,(frmae_width,frame_height))
    # Define the codec and create a VideoWriter object to save the processed video
    out = cv.VideoWriter(os.path.join(baesname, 'Detected_Videos', 'output.avi'),
                                  cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                  (frmae_width, frame_height))
    
    #Loop through the video frames and process time
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        #Now repeat the process for image Object detection 
        image = Image.fromarray(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) #Converting the frame to PIL Image for yolo model
        result = yolo.predict(image)[0]
        
        boxes = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        class_ids = result.boxes.cls.tolist()
        
        frame_output = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            confidence = confidences[i]
            class_id = int(class_ids[i])
            
            label = yolo.names[class_id] if class_id < len(yolo.names) else 'unknown'
            
            frame_output.append(
                {
                    'Class name' : label,
                    'Box' : box,
                    'Confidence Score' : confidence
                }
            )
            
            x, y, w, h = map(int, box)
            color = (255,255,0)
            cv.rectangle(frame,(x,y),(w,h),color,2)
            
            #Adding Label and Confidence Score
            label_text = f'{label} : {confidence }'
            cv.putText(frame,label_text,(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.9,color,2)
            
            
        # Get the video filename without extension (assuming video_file.filename contains the original filename)
        video_filename_without_extension = os.path.splitext(os.path.basename(f.filename))[0]

        # Create a directory for the current video if it doesn't exist
        video_output_folder = os.path.join(baesname, 'Detected_Videos', video_filename_without_extension)
        os.makedirs(video_output_folder, exist_ok=True)

        # Save annotated frame as an image inside the video-specific folder
        frame_filename = os.path.join(video_output_folder, f'frame_{int(cap.get(1))}.jpg')
        cv.imwrite(frame_filename, frame)
        
        #Add frame data to output datalist
        output_data.append(frame_output)

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Prepare the response JSON
    response_data = {
        'frames': output_data
    }

    return jsonify(response_data)
                
    # return jsonify({"File Path":filepath})
@app.route("/",methods=["GET","POST"])
def main():
    result = {}
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

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)