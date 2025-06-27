![image](https://github.com/user-attachments/assets/6fa449d7-01d4-4c81-b6ae-7a3fa6ca021c)

# 🚨 Accident Severity Detection System

An intelligent accident severity detection system using **YOLOv5/YOLOv8**, integrated with **Python**, **Flask**, **MongoDB**, and **Tkinter GUI** for real-time image and video analysis. The system categorizes accidents as **Minor**, **Moderate**, or **Severe**, saves the results to a database, and triggers alerts via **email** and **SMS**.

---

## 📌 Features

- 🔍 **Real-time Detection** from webcam or uploaded images/videos
- 📷 **YOLOv5 & YOLOv8** model support for accurate object detection
- 🗂️ **MongoDB** integration for storing detection logs
- 📤 **Email & SMS alerts** for severe accident cases
- 🧠 Custom-trained model support using `best.pt`
- 🧾 YAML-based class label configuration
- 🖼️ Annotated images/videos with bounding boxes
- 📊 Detection summary with confidence and severity level
- 🎛️ GUI via **Tkinter** and CLI support
- 📁 Result saving and logging with timestamps

---

## 🧰 Tech Stack

| Component        | Technology         |
|------------------|--------------------|
| Framework        | Flask (Backend)    |
| GUI              | Tkinter (Desktop)  |
| Detection Model  | YOLOv5/YOLOv8      |
| Database         | MongoDB            |
| Alerts           | SMTP (Email), HTTP API (SMS) |
| Image Processing | OpenCV             |
| Language         | Python 3.x         |

---

## 📂 Directory Structure
.                                                                                                                                                                                                                   
├── datasets/                                                                                                                                                                                                       
│ └── data.yaml # Class label mapping                                                                                                                                                                               
├── runs/ # YOLO training outputs                                                                                                                                                                                   
├── static/                                                                                                                                                                                                         
│ └── results/ # Output images/videos                                                                                                                                                                               
├── templates/ # Flask HTML templates (if any)                                                                                                                                                                      
├── app.py # Main entry point (Flask + CLI)                                                                                                                                                                         
├── README.md                                                                                                                                                                                                       
├── requirements.txt                                                                                                                                                                                                
└── alert.wav # Sound alert for severe detection                                                                                                                                                                    


---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- MongoDB (local or cloud)
- pip

📦 Install Dependencies
---
```bash
pip install -r requirements.txt
```

Required packages include:
---
torch

opencv-python

pymongo

ultralytics

matplotlib

tkinter (pre-installed with Python)

smtplib, email, requests (standard libs)


⚙️ Setup
---
MongoDB: Ensure MongoDB is running locally (mongodb://localhost:27017/).

Model: Place your YOLOv5 or YOLOv8 model at yolo11n.pt or update the path.

Class Labels: Define labels in datasets/data.yaml as:

names:
  0: Minor
  1: Moderate
  2: Severe

Email Alerts: Update credentials in sendmail() function.

🎮 Run the App
---
✅ GUI Mode (Tkinter)
python app.py

You’ll see a window titled Accident Severity Detection:

Image – Run YOLO on an image

Camera – Detect from webcam in real-time


🧪 Command-Line Interface
---
python app.py --input path/to/image_or_video --model yolo11n.pt

Example:
python app.py --input samples/accident.jpg --model runs/detect/Accident/weights/best.pt


📊 Output
---
📁 Annotated results are saved to static/results/

🧾 Logs are stored in MongoDB:

detections collection

email_logs collection


📤 Severe accidents trigger:
---
Email with image attachment

SMS message with location info

Audio alert via alert.wav


🛠️ Customization
---
🔧 Change Class Names: Update datasets/data.yaml

🧠 Replace Model: Swap out best.pt or yolo11n.pt with your own

📬 Email Receiver: Update toaddr in sendmail() function

📱 SMS API Key: Replace your own key in sendmsg() function

🔐 Security Notes
---
Never hardcode credentials in production. Use environment variables.

Restrict MongoDB access in public deployments.


🤝 Contribution
---
Pull requests and improvements are welcome. Please open issues for bugs or suggestions.

📃 License
MIT License. See LICENSE for more details.

📧 Contact
Developer: VIJAY M
📫 Email: vj17092002@gmail.com
---
