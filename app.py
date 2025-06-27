from flask import Flask, render_template, flash, request, session, jsonify, Response, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import cv2
import numpy as np
import torch
import time
from datetime import datetime
import json
import base64
from pathlib import Path
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import yaml
from werkzeug.utils import secure_filename
from email_utils import send_alert_email
import gridfs
from bson.binary import Binary
from io import BytesIO
import argparse
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'abc'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['accident_severity_db']
users_collection = db['users']
detections_collection = db['detections']
fs = gridfs.GridFS(db)

# Create index for username for faster queries
users_collection.create_index('username', unique=True)

# Initialize the database with admin user if it doesn't exist
if users_collection.count_documents({}) == 0:
    # Insert sample user from the original SQL database
    users_collection.insert_one({
        'name': 'sangeeth Kumar',
        'mobile': '09486365535',
        'email': 'sangeeth5535@gmail.com',
        'address': 'No 16, Samnath Plaza, Madurai Main Road, Melapudhur',
        'username': 'san',
        'password': 'san'
    })

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


def train_model(data_yaml=DATA_YAML_PATH, epochs=100, batch_size=16, img_size=640, weights='yolov8n.pt'):
    """Train a YOLOv8 model on the accident severity dataset"""
    # Create timestamp for run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"accident_severity_{timestamp}"

    # Ensure data_yaml path is correct
    if not os.path.exists(data_yaml):
        print(f"Error: Data file {data_yaml} does not exist")
        return None

    # Load the model
    model = YOLO(weights)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name=run_name,
        patience=20,  # Early stopping patience
        save=True,  # Save best model
        device='0' if torch.cuda.is_available() else 'cpu'
    )

    # Validate the model
    val_results = model.val()

    return {
        'model': model,
        'training_results': results,
        'validation_results': val_results,
        'run_name': run_name
    }


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('Newuser.html')


@app.route("/newuser", methods=['GET', 'POST'])
def Newuser():
    if request.method == 'POST':
        name = request.form['uname']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists!')
            return render_template('Newuser.html')

        # Insert new user into MongoDB
        users_collection.insert_one({
            'name': name,
            'mobile': mobile,
            'email': email,
            'address': address,
            'username': username,
            'password': password
        })

        flash("Record Saved!")
        return render_template('UserLogin.html')

    return render_template('Newuser.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            flash("LOGIN SUCCESSFULLY")

            # Fetch all users from MongoDB
            users = list(users_collection.find())

            return render_template('AdminHome.html', data=users)
        else:
            flash("UserName Or Password Incorrect!")
            return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/Adminhome")
def Adminhomee():
    # Fetch all users from MongoDB
    users = list(users_collection.find())
    return render_template('Adminhome.html', data=users)


# Add these new routes after the existing admin routes

@app.route("/manage_policies")
def manage_policies():
    # Check if admin is logged in (you might want to add a proper admin session check)

    # Fetch all policies from MongoDB
    policies = list(db.insurance_policies.find())

    # Fetch accident statistics for context
    severity_stats = get_severity_statistics()

    return render_template('manage_policies.html', policies=policies, stats=severity_stats)


@app.route("/add_policy", methods=['POST'])
def add_policy():
    if request.method == 'POST':
        name = request.form['name']
        provider = request.form['provider']
        coverage_type = request.form['coverage_type']
        description = request.form['description']
        premium_range = request.form['premium_range']
        recommended_for = request.form.getlist('recommended_for')  # Multiple severity levels
        benefits = request.form['benefits']
        contact_info = request.form['contact_info']

        # Insert new policy into MongoDB
        db.insurance_policies.insert_one({
            'name': name,
            'provider': provider,
            'coverage_type': coverage_type,
            'description': description,
            'premium_range': premium_range,
            'recommended_for': recommended_for,
            'benefits': benefits,
            'contact_info': contact_info,
            'created_at': datetime.now()
        })

        flash("Insurance policy added successfully!")
        return redirect(url_for('manage_policies'))


@app.route("/delete_policy/<policy_id>")
def delete_policy(policy_id):
    try:
        # Delete policy from MongoDB
        result = db.insurance_policies.delete_one({'_id': ObjectId(policy_id)})

        if result.deleted_count > 0:
            flash('Policy deleted successfully!')
        else:
            flash('Policy not found!')
    except Exception as e:
        flash(f'Error deleting policy: {str(e)}')

    return redirect(url_for('manage_policies'))


@app.route("/edit_policy/<policy_id>", methods=['GET', 'POST'])
def edit_policy(policy_id):
    if request.method == 'POST':
        # Update policy in MongoDB
        db.insurance_policies.update_one(
            {'_id': ObjectId(policy_id)},
            {'$set': {
                'name': request.form['name'],
                'provider': request.form['provider'],
                'coverage_type': request.form['coverage_type'],
                'description': request.form['description'],
                'premium_range': request.form['premium_range'],
                'recommended_for': request.form.getlist('recommended_for'),
                'benefits': request.form['benefits'],
                'contact_info': request.form['contact_info'],
                'updated_at': datetime.now()
            }}
        )

        flash("Insurance policy updated successfully!")
        return redirect(url_for('manage_policies'))
    else:
        # Fetch policy for editing
        policy = db.insurance_policies.find_one({'_id': ObjectId(policy_id)})
        return render_template('edit_policy.html', policy=policy)


@app.route("/user_recommendations/<user_id>")
def user_recommendations(user_id):
    # Fetch user's detection history
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    user_detections = list(detections_collection.find({'username': user.get('username', '')}).sort('timestamp', -1))

    # Analyze detection patterns
    severity_counts = {
        'Minor': 0,
        'Moderate': 0,
        'Severe': 0
    }

    for detection in user_detections:
        if 'detections' in detection:
            for det in detection['detections']:
                if 'class_name' in det and det['class_name'] in severity_counts:
                    severity_counts[det['class_name']] += 1

    # Determine predominant severity level
    max_severity = max(severity_counts.items(), key=lambda x: x[1])[0] if any(severity_counts.values()) else None

    # Get recommended policies based on severity pattern
    if max_severity:
        recommended_policies = list(db.insurance_policies.find({'recommended_for': max_severity}))
    else:
        # Default recommendations if no clear pattern
        recommended_policies = list(db.insurance_policies.find().limit(3))

    return render_template('user_recommendations.html',
                           user=user,
                           severity_counts=severity_counts,
                           max_severity=max_severity,
                           policies=recommended_policies)


# Add this new route after the existing user_recommendations route
@app.route("/risk_assessment/<user_id>")
def risk_assessment(user_id):
    # Fetch user information
    user = users_collection.find_one({'_id': ObjectId(user_id)})

    if not user:
        flash('User not found!')
        return redirect(url_for('Adminhomee'))

    # Perform risk assessment based on available data
    # This could include:
    # 1. User's location (some areas have higher accident rates)
    # 2. Frequency of using the detection system (indicates safety consciousness)
    # 3. Types of environments they monitor (construction sites, highways, etc.)

    # For now, we'll use a simple approach based on system usage patterns
    user_detections = list(detections_collection.find({'username': user.get('username', '')}))

    # Calculate risk factors
    detection_frequency = len(user_detections)  # How often they use the system
    detection_recency = 0  # How recently they've used the system

    if user_detections:
        latest_detection = max(user_detections, key=lambda x: x.get('timestamp', datetime.min))
        days_since_last_detection = (datetime.now() - latest_detection.get('timestamp', datetime.now())).days
        detection_recency = max(0, 30 - days_since_last_detection) / 30  # Higher score for more recent usage

    # Calculate risk score (0-100)
    # Higher score = higher risk awareness = lower actual risk
    risk_awareness_score = min(100, (detection_frequency * 5) + (detection_recency * 50))
    risk_level = "Low" if risk_awareness_score > 70 else "Medium" if risk_awareness_score > 40 else "High"

    # Get recommended policies based on risk assessment
    if risk_level == "Low":
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Minor'}))
    elif risk_level == "Medium":
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Moderate'}))
    else:  # High risk
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Severe'}))

    # If no specific recommendations, get general recommendations
    if not recommended_policies:
        recommended_policies = list(db.insurance_policies.find().limit(3))

    return render_template('risk_assessment.html',
                           user=user,
                           risk_level=risk_level,
                           risk_score=risk_awareness_score,
                           policies=recommended_policies)


# Add a new route for users to view their own risk assessment
@app.route("/my_risk_assessment")
def my_risk_assessment():
    # Check if user is logged in
    if 'uname' not in session:
        flash('Please login to view your risk assessment')
        return redirect(url_for('UserLogin'))

    # Find user in MongoDB
    username = session.get('uname', '')
    user = users_collection.find_one({'username': username})

    if not user:
        flash('User not found!')
        return redirect(url_for('UserLogin'))

    # Perform risk assessment
    user_detections = list(detections_collection.find({'username': username}))

    # Calculate risk factors
    detection_frequency = len(user_detections)
    detection_recency = 0

    if user_detections:
        latest_detection = max(user_detections, key=lambda x: x.get('timestamp', datetime.min))
        days_since_last_detection = (datetime.now() - latest_detection.get('timestamp', datetime.now())).days
        detection_recency = max(0, 30 - days_since_last_detection) / 30

    # Calculate risk score
    risk_awareness_score = min(100, (detection_frequency * 5) + (detection_recency * 50))
    risk_level = "Low" if risk_awareness_score > 70 else "Medium" if risk_awareness_score > 40 else "High"

    # Get recommended policies
    if risk_level == "Low":
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Minor'}))
    elif risk_level == "Medium":
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Moderate'}))
    else:
        recommended_policies = list(db.insurance_policies.find({'recommended_for': 'Severe'}))

    if not recommended_policies:
        recommended_policies = list(db.insurance_policies.find().limit(3))

    return render_template('user_risk_assessment.html',
                           user=user,
                           risk_level=risk_level,
                           risk_score=risk_awareness_score,
                           policies=recommended_policies)


# Helper function to get severity statistics
def get_severity_statistics():
    pipeline = [
        {'$unwind': '$detections'},
        {'$group': {
            '_id': '$detections.class_name',
            'count': {'$sum': 1}
        }},
        {'$sort': {'count': -1}}
    ]

    results = list(detections_collection.aggregate(pipeline))
    return results


@app.route("/test_email")
def test_email():
    """Test route to verify email functionality"""
    try:
        # Create a test image
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "test_email_image.jpg")

        # Create a simple test image if it doesn't exist
        if not os.path.exists(test_image_path):
            import numpy as np
            test_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
            # Add some text
            cv2.putText(test_img, "TEST EMAIL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(test_image_path, test_img)

        # Get email from session or use a default
        to_email = session.get('email', 'test@example.com')

        # Send test email
        success, message = send_alert_email(
            to_email,
            "TEST - Accident Detection System",
            test_image_path,
            "Test Severity",
            "Test Location"
        )

        if success:
            flash(f"Test email sent successfully to {to_email}")
        else:
            flash(f"Failed to send test email: {message}")

        return redirect(url_for('AnalyzeImage'))
    except Exception as e:
        flash(f"Error in test email: {str(e)}")
        return redirect(url_for('AnalyzeImage'))


@app.route("/Predict")
def Predict():
    Camera1()

    # Find user in MongoDB
    user = users_collection.find_one({'username': session.get('uname', '')})

    if user:
        return render_template('Userhome.html', data=[user])
    else:
        flash('User not found!')
        return render_template('UserLogin.html')


def Camera1():
    # Load the YOLOv8 model
    model = YOLO('runs/detect/Accident/weights/best.pt')
    # Open the video file
    cap = cv2.VideoCapture(0)
    dd1 = 0
    detection_complete = False
    start_time = time.time()
    max_detection_time = 60  # Maximum time to keep camera open (in seconds)

    # Store detection details for saving to history
    detected_class = None
    detected_confidence = 0
    annotated_frame = None

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        # Check if we've been running too long
        current_time = time.time()
        if current_time - start_time > max_detection_time:
            print("Maximum detection time reached, closing camera")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.6)

        # Always get the latest annotated frame
        annotated_frame = results[0].plot()

        for result in results:
            if result.boxes:
                box = result.boxes[0]
                class_id = int(box.cls)
                object_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"Detected: {object_name} with confidence {confidence:.2f}")

                if object_name != '':
                    dd1 += 1
                    # Store the detection details
                    detected_class = object_name
                    detected_confidence = confidence

                if dd1 >= 40:  # Using the threshold of 40 as requested
                    print(f"Detection threshold reached for {object_name}")
                    detection_complete = True

                    # Save the annotated frame
                    cv2.imwrite("alert.jpg", annotated_frame)

                    # Play alert sound
                    try:
                        import winsound
                        filename = 'alert.wav'
                        winsound.PlaySound(filename, winsound.SND_FILENAME)
                    except Exception as e:
                        print(f"Could not play sound: {str(e)}")

                    # Send notifications
                    try:
                        sendmail()
                        # Use a more generic message that's likely to match approved templates
                        sendmsg(session['mob'], f"Accident Detection Alert: {object_name}")
                        print(f"Notifications sent for {object_name}")
                    except Exception as e:
                        print(f"Error sending notifications: {str(e)}")

                    # Save to detection history
                    try:
                        save_camera_detection_to_history(annotated_frame, object_name, confidence)
                        print("Detection saved to history")
                    except Exception as e:
                        print(f"Error saving to history: {str(e)}")

                    # Break out of the loop to close the camera
                    break

        # Display the annotated frame
        cv2.imshow("Accident Detection", annotated_frame)

        # Break the loop if 'q' is pressed or detection is complete
        if cv2.waitKey(1) & 0xFF == ord("q") or detection_complete:
            break

    # If we exited without completing a detection but have some detection data,
    # save the last good detection to history
    if not detection_complete and detected_class and annotated_frame is not None:
        try:
            save_camera_detection_to_history(annotated_frame, detected_class, detected_confidence)
            print("Last detection saved to history")
        except Exception as e:
            print(f"Error saving last detection to history: {str(e)}")

    # Release the video capture object and close the display window
    print("Closing camera and cleaning up resources")
    cap.release()
    cv2.destroyAllWindows()
    print("Camera detection completed")


def save_camera_detection_to_history(frame, class_name, confidence):
    """
    Save camera detection to the detection history database

    Args:
        frame: The annotated frame with detection visualization
        class_name: The class name of the detected object
        confidence: The confidence score of the detection
    """
    try:
        # Generate filenames with timestamp
        timestamp = int(time.time())
        original_filename = f"camera_{timestamp}.jpg"
        result_filename = f"result_camera_{timestamp}.jpg"

        # Convert frame to bytes for GridFS storage
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("Failed to encode image")
            return

        frame_bytes = buffer.tobytes()

        # Store original and result images in GridFS (same image for camera detection)
        original_image_id = fs.put(
            frame_bytes,
            filename=original_filename,
            content_type='image/jpeg',
            upload_date=datetime.now()
        )

        result_image_id = fs.put(
            frame_bytes,
            filename=result_filename,
            content_type='image/jpeg',
            upload_date=datetime.now()
        )

        # Create detection data
        detections = [{
            'class_name': class_name,
            'confidence': confidence
        }]

        # Determine if this is a severe detection
        is_severe = (class_name == 'Severe')

        # Save to MongoDB
        detection_data = {
            'original_image_id': original_image_id,
            'result_image_id': result_image_id,
            'original_filename': original_filename,
            'result_filename': result_filename,
            'timestamp': datetime.now(),
            'username': session.get('uname', 'anonymous'),
            'detections': detections,
            'is_alert': is_severe,
            'detection_method': 'camera',  # Add this to distinguish from uploaded images
            'detection_time': timestamp
        }

        # Insert into detection collection
        detection_id = detections_collection.insert_one(detection_data).inserted_id
        print(f"Camera detection saved to history with ID: {detection_id}")

        return detection_id

    except Exception as e:
        print(f"Error saving camera detection to history: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def sendmsg(targetno, message):
    """
    Send SMS notification using the SMS API

    Args:
        targetno (str): Target mobile number
        message (str): Message content

    Returns:
        bool: Success status
    """
    try:
        print(f"Sending SMS to {targetno}: {message}")

        # Extract the severity level from the message
        severity = "accident"
        if "Severe" in message:
            severity = "severe accident"
        elif "Moderate" in message:
            severity = "moderate accident"
        elif "Minor" in message:
            severity = "minor accident"

        # Use a pre-approved template format
        template_message = f"ALERT: A {severity} has been detected by your Accident Detection System. Please check your app for details."

        # Format the message for URL encoding
        import urllib.parse
        encoded_message = urllib.parse.quote(template_message)

        # Send the SMS request
        response = requests.post(
            f"http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno={targetno}&text={encoded_message}")

        # Check if the request was successful
        if response.status_code == 200:
            print(f"SMS sent successfully to {targetno}")
            return True
        else:
            print(f"SMS API error: {response.status_code} - {response.text}")
            # Try an alternative template if the first one fails
            alt_template = f"Safety Alert: Please check your Accident Detection app for important updates regarding a recent detection."
            encoded_alt = urllib.parse.quote(alt_template)

            alt_response = requests.post(
                f"http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno={targetno}&text={encoded_alt}")

            if alt_response.status_code == 200:
                print(f"Alternative SMS template sent successfully to {targetno}")
                return True
            else:
                print(f"Alternative SMS template also failed: {alt_response.status_code} - {alt_response.text}")
                return False

    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False


def sendmail():
    fromaddr = "projectmailm@gmail.com"
    toaddr = session['email']

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = "Accident Severity Detection"

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


@app.route("/delete_user/<user_id>")
def delete_user(user_id):
    try:
        # Delete user from MongoDB
        result = users_collection.delete_one({'_id': ObjectId(user_id)})

        if result.deleted_count > 0:
            flash('User deleted successfully!')
        else:
            flash('User not found!')
    except Exception as e:
        flash(f'Error deleting user: {str(e)}')

    # Redirect to admin home
    return Adminhomee()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file
        filename = f"upload_{int(time.time())}{Path(file.filename).suffix}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        results = process_image(filepath)

        return jsonify({
            'filename': filename,
            'filepath': filepath,
            'results': results
        })


def process_image(image_path):
    try:
        # Load model
        model = load_model()

        # Perform inference
        results = model(image_path)

        # Process results
        detections = []
        for pred in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            detections.append({
                'class': int(cls),
                'class_name': class_names[int(cls)],
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

        # Save the result image with bounding boxes
        result_img = results.render()[0]
        result_filename = f"result_{Path(image_path).name}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)

        return {
            'detections': detections,
            'result_image': f"/static/uploads/{result_filename}"
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return {'error': str(e)}


@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    """Generate frames from webcam with accident detection"""
    cap = cv2.VideoCapture(0)
    model = load_model()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame with YOLO
        results = model(frame)
        result_frame = results.render()[0]

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded video
        filename = f"video_{int(time.time())}{Path(file.filename).suffix}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the video (just analyze first frame for demo)
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Could not read video file'})

        # Save first frame as image
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{filename}.jpg")
        cv2.imwrite(frame_path, frame)

        # Process the frame
        results = process_image(frame_path)

        return jsonify({
            'filename': filename,
            'filepath': filepath,
            'frame_results': results
        })


def load_model():
    global model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo11n.pt')
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45  # IoU threshold
    return model


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in MongoDB
        user = users_collection.find_one({'username': username, 'password': password})

        if user is None:
            flash('Username or Password is wrong')
            return render_template('Userlogin.html')
        else:
            # Store user info in session
            session['uname'] = username
            session['mob'] = user.get('mobile', '')
            session['email'] = user.get('email', '')

            flash("LOGIN SUCCESSFULLY")
            return redirect(url_for('Userhome'))


@app.route("/Userhome")
def Userhome():
    # Find user in MongoDB
    user = users_collection.find_one({'username': session.get('uname', '')})

    if user:
        return render_template('Userhome.html', data=[user])
    else:
        flash('User not found!')
        return render_template('UserLogin.html')


# Add a new route for detection history after the Userhome route
@app.route("/DetectionHistory")
def DetectionHistory():
    # Check if user is logged in
    if 'uname' not in session:
        flash('Please login to view detection history')
        return redirect(url_for('UserLogin'))

    # Fetch detection records for the current user from MongoDB
    username = session.get('uname', '')
    detections = list(detections_collection.find(
        {'username': username}
    ).sort('timestamp', -1))  # Sort by timestamp descending (newest first)

    # Find user in MongoDB
    user = users_collection.find_one({'username': username})

    if not user:
        flash('User not found!')
        return redirect(url_for('UserLogin'))

    return render_template('detection_history.html',
                           detections=detections,
                           user=user)


# Add routes for viewing and deleting individual detection records
@app.route("/ViewDetection/<detection_id>")
def ViewDetection(detection_id):
    # Check if user is logged in
    if 'uname' not in session:
        flash('Please login to view detection details')
        return redirect(url_for('UserLogin'))

    try:
        # Fetch the detection record from MongoDB
        detection = detections_collection.find_one({'_id': ObjectId(detection_id)})

        if not detection:
            flash('Detection record not found')
            return redirect(url_for('DetectionHistory'))

        # Check if this detection belongs to the current user
        if detection.get('username') and detection.get('username') != session.get('uname'):
            flash('You do not have permission to view this detection')
            return redirect(url_for('DetectionHistory'))

        # Check if severe accident was detected
        has_severe = False
        if detection.get('detections'):
            has_severe = any(d.get('class_name') == 'Severe' for d in detection['detections'])

        # Convert ObjectId to string if they exist
        if 'original_image_id' in detection and not isinstance(detection['original_image_id'], str):
            detection['original_image_id'] = str(detection['original_image_id'])

        if 'result_image_id' in detection and not isinstance(detection['result_image_id'], str):
            detection['result_image_id'] = str(detection['result_image_id'])

        return render_template('view_detection.html',
                               detection=detection,
                               has_severe=has_severe)

    except Exception as e:
        flash(f'Error retrieving detection: {str(e)}')
        return redirect(url_for('DetectionHistory'))


@app.route("/DeleteDetection/<detection_id>")
def DeleteDetection(detection_id):
    # Check if user is logged in
    if 'uname' not in session:
        flash('Please login to delete detection records')
        return redirect(url_for('UserLogin'))

    try:
        # Fetch the detection record from MongoDB
        detection = detections_collection.find_one({'_id': ObjectId(detection_id)})

        if not detection:
            flash('Detection record not found')
            return redirect(url_for('DetectionHistory'))

        # Check if this detection belongs to the current user
        if detection.get('username') and detection.get('username') != session.get('uname'):
            flash('You do not have permission to delete this detection')
            return redirect(url_for('DetectionHistory'))

        # Delete the detection record
        result = detections_collection.delete_one({'_id': ObjectId(detection_id)})

        if result.deleted_count > 0:
            # Try to delete the associated image files if they exist
            try:
                if detection.get('image_path') and os.path.exists(detection['image_path']):
                    os.remove(detection['image_path'])

                if detection.get('result_path') and os.path.exists(detection['result_path']):
                    os.remove(detection['result_path'])
            except Exception as e:
                print(f"Error deleting image files: {str(e)}")

            flash('Detection record deleted successfully')
        else:
            flash('Failed to delete detection record')

        return redirect(url_for('DetectionHistory'))

    except Exception as e:
        flash(f'Error deleting detection: {str(e)}')
        return redirect(url_for('DetectionHistory'))


@app.route("/AnalyzeImage", methods=['GET', 'POST'])
def AnalyzeImage():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Read the file data
            file_data = file.read()

            # Generate filenames
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            new_filename = f"upload_{timestamp}_{filename}"

            # Store original image in GridFS
            original_image_id = fs.put(
                file_data,
                filename=new_filename,
                content_type=file.content_type,
                upload_date=datetime.now()
            )

            # Process the image with YOLO
            try:
                # Convert bytes to numpy array for OpenCV
                nparr = np.frombuffer(file_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Find and load model
                model_path = 'runs/detect/Accident/weights/best.pt'
                if not os.path.exists(model_path):
                    alt_paths = ['yolo11n.pt', 'runs/detect/crack/weights/best.pt']
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            break
                    else:
                        flash("Error: No model file found.")
                        return redirect(request.url)

                # Load model and run inference
                model = YOLO(model_path)
                results = model(img, conf=0.4)

                # Get the annotated image
                annotated_image = results[0].plot()

                # Convert annotated image to bytes
                result_filename = f"result_{timestamp}_{filename}"
                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                result_bytes = buffer.tobytes()

                # Store result image in GridFS
                result_image_id = fs.put(
                    result_bytes,
                    filename=result_filename,
                    content_type='image/jpeg',
                    upload_date=datetime.now()
                )

                # Extract detection information
                detections = []
                has_severe = False

                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            object_name = model.names[class_id]
                            confidence = float(box.conf[0])

                            if object_name == 'Severe':
                                has_severe = True

                            detections.append({
                                'class_name': object_name,
                                'confidence': confidence
                            })

                # Save to MongoDB
                detection_data = {
                    'original_image_id': original_image_id,
                    'result_image_id': result_image_id,
                    'original_filename': new_filename,
                    'result_filename': result_filename,
                    'timestamp': datetime.now(),
                    'username': session.get('uname', 'anonymous'),
                    'detections': detections,
                    'is_alert': has_severe
                }

                detection_id = detections_collection.insert_one(detection_data).inserted_id

                # Send notifications for all analyses (not just severe)
                print(f"Analysis completed - Preparing to send alerts")

                # Determine the highest severity level detected
                severity_level = "None"
                if detections:
                    severity_levels = [d.get('class_name', 'None') for d in detections]
                    if 'Severe' in severity_levels:
                        severity_level = "Severe"
                    elif 'Moderate' in severity_levels:
                        severity_level = "Moderate"
                    elif 'Minor' in severity_levels:
                        severity_level = "Minor"
                else:
                    severity_level = "None"

                # Save result image temporarily for email attachment
                temp_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{result_filename}")
                try:
                    with open(temp_result_path, 'wb') as f:
                        f.write(result_bytes)
                    print(f"Temporary result image saved to: {temp_result_path}")

                    # Verify the file exists
                    if not os.path.exists(temp_result_path):
                        print(f"ERROR: Temporary file was not created at {temp_result_path}")
                except Exception as e:
                    print(f"Error saving temporary result image: {str(e)}")
                    temp_result_path = None

                # Send SMS notification
                if 'mob' in session and session['mob']:
                    try:
                        # Customize message based on severity
                        if severity_level == "Severe":
                            sms_message = "URGENT: Severe accident detected in your uploaded image!"
                        elif severity_level == "Moderate":
                            sms_message = "ALERT: Moderate accident detected in your uploaded image."
                        elif severity_level == "Minor":
                            sms_message = "NOTICE: Minor accident detected in your uploaded image."
                        else:
                            sms_message = "Analysis completed for your uploaded image."

                        sendmsg(session['mob'], sms_message)
                        print(f"SMS alert sent to {session['mob']}")
                        flash(f"SMS notification sent to your registered mobile number")
                    except Exception as e:
                        print(f"Error sending SMS: {str(e)}")
                        flash(f"Could not send SMS notification: {str(e)}")

                # Send email notification
                if 'email' in session and session['email']:
                    try:
                        print(f"Attempting to send email alert to: {session['email']}")

                        # Customize email subject based on severity
                        if severity_level == "Severe":
                            email_subject = "URGENT: Severe Accident Detected - Immediate Attention Required"
                        elif severity_level == "Moderate":
                            email_subject = "ALERT: Moderate Accident Detected in Your Image"
                        elif severity_level == "Minor":
                            email_subject = "NOTICE: Minor Accident Detected in Your Image"
                        else:
                            email_subject = "Image Analysis Results - Accident Detection System"

                        # Send email with appropriate subject and severity level
                        if temp_result_path and os.path.exists(temp_result_path):
                            email_success, email_message = send_alert_email(
                                session['email'],
                                email_subject,
                                temp_result_path,
                                severity_level,
                                "Uploaded Image Analysis"
                            )

                            if email_success:
                                print(f"Email alert successfully sent to {session['email']}")
                                flash(f"Analysis results sent to {session['email']}")
                            else:
                                print(f"Failed to send email alert: {email_message}")
                                flash(f"Note: We couldn't send an email alert. Please check your email settings.")
                        else:
                            print(f"Cannot send email - temporary file not available")
                            flash("Analysis email could not be sent - image processing issue")
                    except Exception as e:
                        print(f"Unexpected error in email sending process: {str(e)}")
                        flash("There was a problem sending the analysis email")

                # Clean up temporary file
                try:
                    if temp_result_path and os.path.exists(temp_result_path):
                        os.remove(temp_result_path)
                        print(f"Temporary file removed: {temp_result_path}")
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")

                # Create URLs for the images
                original_image_url = url_for('get_image', file_id=str(original_image_id))
                result_image_url = url_for('get_image', file_id=str(result_image_id))

                return render_template('analysis_result.html',
                                       image_path=original_image_url,
                                       result_path=result_image_url,
                                       detections=detections,
                                       has_severe=has_severe,
                                       detection_id=str(detection_id))

            except Exception as e:
                import traceback
                traceback.print_exc()
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed')
            return redirect(request.url)

    return render_template('upload_image.html')


# Add a route to retrieve images from GridFS
@app.route('/get_image/<file_id>')
def get_image(file_id):
    try:
        # Find the file in GridFS
        file_data = fs.get(ObjectId(file_id))

        # Return the file as a response
        return Response(
            file_data.read(),
            mimetype=file_data.content_type,
            headers={"Content-Disposition": f"inline; filename={file_data.filename}"}
        )
    except Exception as e:
        return f"Error retrieving image: {str(e)}", 404


def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Replace the sendmail_with_image function with this improved version:

def sendmail_with_image(image_path):
    """Send email with the specified image"""
    if 'email' not in session:
        print("No email in session, cannot send email")
        return False, "No email in session"

    to_email = session['email']
    severity_level = "Severe"  # You can make this dynamic based on detection
    location = "Uploaded Image Analysis"

    print(f"Sending direct email to {to_email} with image: {image_path}")

    try:
        fromaddr = "projectmailm@gmail.com"
        password = "qmgn xecl bkqv musr"  # App password for Gmail

        # Create message
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = to_email
        msg['Subject'] = "URGENT: Severe Accident Detected"

        # Email body
        body = """
        <html>
        <body>
            <h2 style="color: #d9534f;">SEVERE ACCIDENT DETECTED</h2>
            <p>A severe accident has been detected in your uploaded image.</p>
            <p>Please check the attached image for details.</p>
            <p>This is an automated alert from the Accident Severity Detection System.</p>
        </body>
        </html>
        """

        # Attach HTML content
        msg.attach(MIMEText(body, 'html'))

        # Attach image if it exists
        if image_path and os.path.isfile(image_path):
            print(f"Attaching image: {image_path}")
            with open(image_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                filename = os.path.basename(image_path)
                part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(part)
        else:
            print(f"Image path not found or invalid: {image_path}")

        # Connect to SMTP server
        print("Connecting to SMTP server...")
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(fromaddr, password)

        # Send email
        print("Sending email...")
        s.send_message(msg)
        s.quit()

        print("Direct email sent successfully")
        return True, "Email sent successfully"

    except Exception as e:
        print(f"Error in direct email sending: {str(e)}")
        return False, str(e)


@app.route("/admin_detection_stats")
def admin_detection_stats():
    # Check if admin is logged in (you should implement proper admin authentication)

    # Get detection statistics
    stats = {}

    # Total detections
    stats['total_detections'] = detections_collection.count_documents({})

    # Detections by severity
    severity_pipeline = [
        {'$unwind': '$detections'},
        {'$group': {
            '_id': '$detections.class_name',
            'count': {'$sum': 1}
        }},
        {'$sort': {'count': -1}}
    ]
    stats['severity_counts'] = list(detections_collection.aggregate(severity_pipeline))

    # Detections by day (last 7 days)
    from datetime import timedelta
    seven_days_ago = datetime.now() - timedelta(days=7)

    daily_pipeline = [
        {'$match': {'timestamp': {'$gte': seven_days_ago}}},
        {'$group': {
            '_id': {
                'year': {'$year': '$timestamp'},
                'month': {'$month': '$timestamp'},
                'day': {'$dayOfMonth': '$timestamp'}
            },
            'count': {'$sum': 1}
        }},
        {'$sort': {'_id.year': 1, '_id.month': 1, '_id.day': 1}}
    ]
    stats['daily_counts'] = list(detections_collection.aggregate(daily_pipeline))

    # Format dates for chart
    chart_labels = []
    chart_data = []

    for i in range(7):
        date = datetime.now() - timedelta(days=6 - i)
        formatted_date = date.strftime('%Y-%m-%d')
        chart_labels.append(formatted_date)

        # Find matching count
        count = 0
        for entry in stats['daily_counts']:
            entry_date = datetime(
                entry['_id']['year'],
                entry['_id']['month'],
                entry['_id']['day']
            )
            if entry_date.strftime('%Y-%m-%d') == formatted_date:
                count = entry['count']
                break

        chart_data.append(count)

    stats['chart_labels'] = chart_labels
    stats['chart_data'] = chart_data

    # Recent detections
    stats['recent_detections'] = list(detections_collection.find().sort('timestamp', -1).limit(10))

    # User statistics
    stats['total_users'] = users_collection.count_documents({})
    stats['active_users'] = len(detections_collection.distinct('username'))

    return render_template('admin_detection_stats.html', stats=stats)


@app.route("/admin_settings", methods=['GET', 'POST'])
def admin_settings():
    # Check if admin is logged in

    # Get current settings
    settings = db.settings.find_one({'_id': 'global_settings'})
    if not settings:
        # Create default settings if they don't exist
        settings = {
            '_id': 'global_settings',
            'detection_threshold': 0.4,
            'alert_threshold': 0.6,
            'enable_email_alerts': True,
            'enable_sms_alerts': True,
            'admin_email': 'admin@example.com',
            'system_name': 'Accident Severity Detection System'
        }
        db.settings.insert_one(settings)

    if request.method == 'POST':
        # Update settings
        updated_settings = {
            'detection_threshold': float(request.form.get('detection_threshold', 0.4)),
            'alert_threshold': float(request.form.get('alert_threshold', 0.6)),
            'enable_email_alerts': request.form.get('enable_email_alerts') == 'on',
            'enable_sms_alerts': request.form.get('enable_sms_alerts') == 'on',
            'admin_email': request.form.get('admin_email', 'admin@example.com'),
            'system_name': request.form.get('system_name', 'Accident Severity Detection System')
        }

        db.settings.update_one(
            {'_id': 'global_settings'},
            {'$set': updated_settings}
        )

        flash('Settings updated successfully!')
        return redirect(url_for('admin_settings'))

    return render_template('admin_settings.html', settings=settings)


if __name__ == '__main__':
    model = None

    # Check if running as a script or as a web app
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        main()
    else:
        app.run(debug=True, use_reloader=True)
