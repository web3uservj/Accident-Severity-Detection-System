import os
import cv2
import torch
import numpy as np
import time
from datetime import datetime
import argparse
from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
from tkinter import messagebox
from pymongo import MongoClient
import winsound
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import yaml
import sys

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['accident_severity_db']
detections_collection = db['detections']

# Path to data.yaml file
DATA_YAML_PATH = os.path.join('datasets', 'data.yaml')


# Load class names from data.yaml
def load_class_names():
    if os.path.exists(DATA_YAML_PATH):
        with open(DATA_YAML_PATH, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', ['Minor', 'Moderate', 'Severe'])
    return ['Minor', 'Moderate', 'Severe']  # Default if file not found


# Class names from your dataset
class_names = load_class_names()


def load_model(model_path='yolo11n.pt'):
    """Load YOLOv11 model"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45  # IoU threshold
    return model


def process_image(model, image_path, output_dir='static/results'):
    """Process a single image with the model"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Perform inference
    results = model(img)

    # Process results
    detections = []

    for pred in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = pred
        class_idx = int(cls)
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class_{class_idx}"

        detections.append({
            'class': class_idx,
            'class_name': class_name,
            'confidence': float(conf),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })

    # Save the result image with bounding boxes
    result_img = results.render()[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"result_{timestamp}_{os.path.basename(image_path)}")
    cv2.imwrite(result_path, result_img)

    # Check for severe accidents
    has_severe = any(d['class_name'] == 'Severe' for d in detections)

    # Save detections to MongoDB
    if detections:
        save_detection_to_mongodb(image_path, detections)

    return {
        'detections': detections,
        'result_image': result_path,
        'has_severe': has_severe
    }


def process_video(model, video_path, output_dir='static/results'):
    """Process a video with the model"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"result_{timestamp}_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    severe_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = model(frame)

        # Check for severe accidents
        for pred in results.xyxy[0].cpu().numpy():
            cls = int(pred[5])
            if cls == 2:  # Severe class
                severe_frames += 1
                break

        # Render frame with detections
        result_frame = results.render()[0]

        # Write frame to output video
        out.write(result_frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()

    return {
        'output_video': output_path,
        'frames_processed': frame_count,
        'severe_frames': severe_frames,
        'severe_percentage': (severe_frames / frame_count * 100) if frame_count > 0 else 0
    }


def save_detection_to_mongodb(image_path, detections):
    """Save detection results to MongoDB"""
    detection_data = {
        'image_path': image_path,
        'timestamp': datetime.now(),
        'detections': []
    }

    for detection in detections:
        detection_data['detections'].append({
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'bbox': detection['bbox']
        })

    # Insert into MongoDB
    result = detections_collection.insert_one(detection_data)
    print(f"Detection saved to MongoDB with ID: {result.inserted_id}")


def imgtest():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    from ultralytics import YOLO

    # Try to find the model file
    model_path = 'runs/detect/crack/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Looking for alternatives...")
        alt_paths = [
            'yolo11n.pt',
            'runs/detect/Accident/weights/best.pt'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative model: {model_path}")
                break
        else:
            print("Error: No model file found. Please check your model paths.")
            return

    model = YOLO(model_path)

    # Run YOLOv8 inference on the image
    results = model(image, conf=0.1)

    # Annotate the image
    annotated_image = image.copy()

    detections = []

    for result in results:
        if result.boxes:
            for box in result.boxes:
                # Extract class ID and name
                class_id = int(box.cls)
                object_name = model.names[class_id]

                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract confidence
                confidence = float(box.conf[0])

                # Calculate bounding box area
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Draw the bounding box and label
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{object_name}"
                cv2.putText(
                    annotated_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                print(f"Detected: {object_name}")

                # Add detection to list
                detections.append({
                    'class_name': object_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

    # Save detections to MongoDB
    if detections:
        save_detection_to_mongodb(import_file_path, detections)

    # Display the annotated image
    cv2.imshow("YOLOv8 Prediction", annotated_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def Camera1():
    import cv2
    from ultralytics import YOLO
    from datetime import datetime

    # Try to find the model file
    model_path = 'runs/detect/Accident/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Looking for alternatives...")
        alt_paths = [
            'yolo11n.pt',
            'runs/detect/crack/weights/best.pt'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative model: {model_path}")
                break
        else:
            print("Error: No model file found. Please check your model paths.")
            return

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(0)
    dd1 = 0

    # For MongoDB storage
    session_start_time = datetime.now()
    session_detections = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.4)

            frame_detections = []

            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        object_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        print(object_name)

                        # Add to frame detections
                        frame_detections.append({
                            'class_name': object_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now()
                        })

                        if object_name != '':
                            dd1 += 1

                        if dd1 == 20:
                            dd1 = 0
                            filename = 'alert.wav'
                            winsound.PlaySound(filename, winsound.SND_FILENAME)

                            annotated_frame = results[0].plot()
                            cv2.imwrite("alert.jpg", annotated_frame)

                            # Save this detection to MongoDB
                            alert_detection = {
                                'image_path': 'alert.jpg',
                                'timestamp': datetime.now(),
                                'detections': frame_detections,
                                'is_alert': True
                            }
                            detections_collection.insert_one(alert_detection)

                            sendmail()
                            sendmsg("9486365535", "Location CARE College Prediction Name:" + object_name)

            # Add frame detections to session detections
            if frame_detections:
                session_detections.extend(frame_detections)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Save session data to MongoDB
    if session_detections:
        session_data = {
            'session_start': session_start_time,
            'session_end': datetime.now(),
            'detections': session_detections
        }
        detections_collection.insert_one(session_data)

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def sendmsg(targetno, message):
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


def sendmail():
    fromaddr = "projectmailm@gmail.com"
    toaddr = "sangeeth5535@gmail.com"

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = "Crack Detection"

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    filename = "alert.jpg"
    attachment = open("alert.jpg", "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "qmgn xecl bkqv musr")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()

    # Log email in MongoDB
    email_log = {
        'timestamp': datetime.now(),
        'to_email': toaddr,
        'subject': 'Alert',
        'attachment': 'alert.jpg'
    }
    db.email_logs.insert_one(email_log)


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    main_screen.configure()
    main_screen.title("Accident Severity Detection   ")

    Label(text=" Accident Severity Detection  ", width="300", height="5", font=("Calibri", 16)).pack()

    Label(text="").pack()
    Button(text="Image", font=(
        'Verdana', 15), height="2", width="30", command=imgtest).pack(side=TOP)
    Label(text="").pack()
    Button(text="Camera", font=(
        'Verdana', 15), height="2", width="30", command=Camera1).pack(side=TOP)

    main_screen.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Accident Severity Detection')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to YOLOv11 model')
    parser.add_argument('--output', type=str, default='static/results', help='Output directory')
    parser.add_argument('--data', type=str, default=DATA_YAML_PATH, help='Path to data.yaml file')
    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Warning: Data file {args.data} does not exist. This may affect class name mapping.")

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return

    # Load model
    print("Loading model...")
    model = load_model(args.model)

    # Process input based on file type
    file_ext = os.path.splitext(args.input)[1].lower()

    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"Processing image: {args.input}")
        result = process_image(model, args.input, args.output)

        if result:
            print(f"Results saved to: {result['result_image']}")
            print(f"Detections: {len(result['detections'])}")

            for i, det in enumerate(result['detections']):
                print(f"  {i + 1}. {det['class_name']} ({det['confidence']:.2f})")

            if result['has_severe']:
                print("WARNING: Severe accident detected!")

    elif file_ext in ['.mp4', '.avi', '.mov']:
        print(f"Processing video: {args.input}")
        result = process_video(model, args.input, args.output)

        if result:
            print(f"Results saved to: {result['output_video']}")
            print(f"Frames processed: {result['frames_processed']}")
            print(f"Frames with severe accidents: {result['severe_frames']} ({result['severe_percentage']:.2f}%)")

            if result['severe_frames'] > 0:
                print("WARNING: Severe accidents detected in video!")

    else:
        print(f"Error: Unsupported file type: {file_ext}")


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        main_account_screen()
