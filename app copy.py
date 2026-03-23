# import mysql.connector, os, re
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import uuid
# import matplotlib.pyplot as plt
# import pymysql
# from flask import Flask, session, render_template, request, redirect, url_for
# from timm import create_model
# import os


# app = Flask(__name__)
# app.secret_key = 'admin'

# mydb = pymysql.connect(
#     host="localhost",
#     user="root",
#     password="root",
#     port=3306,
#     database='Under'
# )

# mycursor = mydb.cursor()

# def executionquery(query,values):
#     mycursor.execute(query,values)
#     mydb.commit()
#     return

# def retrivequery1(query,values):
#     mycursor.execute(query,values)
#     data = mycursor.fetchall()
#     return data

# def retrivequery2(query):
#     mycursor.execute(query)
#     data = mycursor.fetchall()
#     return data

# @app.route('/')
# def index():
#     return render_template('index.html')



# @app.route('/register', methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#         # Retrieve form data
#         name = request.form['name']  # Added name field
#         email = request.form['email']
#         password = request.form['password']
#         c_password = request.form['c_password']
        
#         # Check if passwords match
#         if password == c_password:
#             # Query to check if the email already exists (case-insensitive)
#             query = "SELECT UPPER(email) FROM users"
#             email_data = retrivequery2(query)
#             email_data_list = [i[0] for i in email_data]
            
#             # If the email is unique, insert the new user
#             if email.upper() not in email_data_list:
#                 query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
#                 values = (name, email, password)  # Include name in the insert query
#                 executionquery(query, values)
#                 return render_template('register.html', message="Successfully Registered!")
            
#             # If email already exists
#             return render_template('register.html', message="This email ID already exists!")
        
#         # If passwords do not match
#         return render_template('register.html', message="Confirm password does not match!")
    
#     # If GET request
#     return render_template('register.html')


# @app.route('/login', methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form['email']
#         password = request.form['password']
        
#         # Query to check if email exists
#         query = "SELECT UPPER(email) FROM users"
#         email_data = retrivequery2(query)
#         email_data_list = [i[0] for i in email_data]  # Simplified list comprehension

#         if email.upper() in email_data_list:
#             # Query to fetch the password for the provided email
#             query = "SELECT password, name FROM users WHERE email = %s"
#             values = (email,)
#             user_data = retrivequery1(query, values)  # Assuming this returns a list of tuples
            
#             if user_data:
#                 stored_password, name = user_data[0]  # Extract the password and name
                
#                 # Check if password matches (case-insensitive)
#                 if password == stored_password:
#                     # Store the email and name in a session or global variable
#                     session['user_email'] = email  # Store in session for security
#                     session['user_name'] = name
                    
#                     # Pass the user's name to the home page directly
#                     return render_template('home.html', user_name=name)  # Pass the user name to home page
                
#                 # If passwords do not match
#                 return render_template('login.html', message="Invalid Password!")
            
#             # If no data found for the user (which shouldn't happen here)
#             return render_template('login.html', message="This email ID does not exist!")
        
#         # If email doesn't exist
#         return render_template('login.html', message="This email ID does not exist!")
    
#     # If GET request
#     return render_template('login.html')


# @app.route('/home')
# def home():
#     # Check if user is logged in by verifying session
#     if 'user_email' not in session:
#         return redirect(url_for('login'))  # Redirect to login page if not logged in
    
#     user_name = session.get('user_name')  # Retrieve user name from session
#     return render_template('home.html', user_name=user_name)


# @app.route('/about')
# def about():
#     return render_template('about.html')




# # Model performance reports (hardcoded from your provided results)
# MODEL_REPORTS = {
#     "mobilenet": {
#         "classes": {
#             "blue_tint": {"precision": 0.94, "recall": 0.97, "f1-score": 0.96, "support": 186},
#             "blurry": {"precision": 0.99, "recall": 0.86, "f1-score": 0.92, "support": 174},
#             "green_tint": {"precision": 0.88, "recall": 0.99, "f1-score": 0.93, "support": 160},
#             "hazy": {"precision": 1.00, "recall": 0.99, "f1-score": 1.00, "support": 176},
#             "high_contrast": {"precision": 0.87, "recall": 0.78, "f1-score": 0.82, "support": 175},
#             "low_illumination": {"precision": 0.91, "recall": 0.98, "f1-score": 0.94, "support": 183},
#             "noisy": {"precision": 1.00, "recall": 0.99, "f1-score": 1.00, "support": 182},
#             "raw-890": {"precision": 0.77, "recall": 0.77, "f1-score": 0.77, "support": 184},
#             "red_tint": {"precision": 0.98, "recall": 0.98, "f1-score": 0.98, "support": 185},
#         },
#         "accuracy": 0.93,
#         "macro_avg": {"precision": 0.93, "recall": 0.93, "f1-score": 0.92},
#         "weighted_avg": {"precision": 0.93, "recall": 0.93, "f1-score": 0.92}
#     },
#     "resnet": {
#         "classes": {
#             "blue_tint": {"precision": 0.98, "recall": 0.96, "f1-score": 0.97, "support": 167},
#             "blurry": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92, "support": 185},
#             "green_tint": {"precision": 0.94, "recall": 0.98, "f1-score": 0.96, "support": 192},
#             "hazy": {"precision": 0.98, "recall": 1.00, "f1-score": 0.99, "support": 184},
#             "high_contrast": {"precision": 0.88, "recall": 0.78, "f1-score": 0.83, "support": 174},
#             "low_illumination": {"precision": 0.90, "recall": 0.98, "f1-score": 0.94, "support": 192},
#             "noisy": {"precision": 0.99, "recall": 1.00, "f1-score": 1.00, "support": 160},
#             "raw-890": {"precision": 0.75, "recall": 0.69, "f1-score": 0.72, "support": 172},
#             "red_tint": {"precision": 0.96, "recall": 0.98, "f1-score": 0.97, "support": 179},
#         },
#         "accuracy": 0.92,
#         "macro_avg": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92},
#         "weighted_avg": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92}
#     },
#     "cnn": {
#         "classes": {
#             "blue_tint": {"precision": 0.9532, "recall": 0.9645, "f1-score": 0.9588, "support": 169},
#             "blurry": {"precision": 0.8901, "recall": 0.8571, "f1-score": 0.8733, "support": 189},
#             "green_tint": {"precision": 0.9529, "recall": 1.0000, "f1-score": 0.9759, "support": 162},
#             "hazy": {"precision": 0.9692, "recall": 1.0000, "f1-score": 0.9844, "support": 189},
#             "high_contrast": {"precision": 0.9375, "recall": 0.7459, "f1-score": 0.8308, "support": 181},
#             "low_illumination": {"precision": 0.9109, "recall": 0.9946, "f1-score": 0.9509, "support": 185},
#             "noisy": {"precision": 1.0000, "recall": 0.9946, "f1-score": 0.9973, "support": 184},
#             "raw-890": {"precision": 0.7513, "recall": 0.8256, "f1-score": 0.7867, "support": 172},
#             "red_tint": {"precision": 0.9586, "recall": 0.9310, "f1-score": 0.9446, "support": 174},
#         },
#         "accuracy": 0.9234,
#         "macro_avg": {"precision": 0.9249, "recall": 0.9237, "f1-score": 0.9225},
#         "weighted_avg": {"precision": 0.9253, "recall": 0.9234, "f1-score": 0.9226}
#     }
# }

# # Now update the /model route to pass the selected report
# @app.route('/model', methods=['GET', 'POST'])
# def model():
#     if 'user_email' not in session:
#         return redirect(url_for('login'))

#     selected_model = None
#     report = None

#     if request.method == 'POST':
#         selected_model = request.form.get('modelSelect')
#         if selected_model in MODEL_REPORTS:
#             report = MODEL_REPORTS[selected_model]

#     return render_template('model.html', 
#                           selected_model=selected_model, 
#                           report=report,
#                           models_available=["mobilenet", "resnet", "cnn"])



# # Upload folder
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ────────────────────────────────────────────────────────────────
# #  MODEL SETUP - Underwater Image Degradation Classification
# # ────────────────────────────────────────────────────────────────

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # IMPORTANT: These must match EXACTLY what you used during training
# # (usually printed as dataset.classes or you can hardcode them)
# CLASSES = ['blue_tint',
#  'blurry',
#  'green_tint',
#  'hazy',
#  'high_contrast',
#  'low_illumination',
#  'noisy',
#  'raw-890',
#  'red_tint']

# NUM_CLASSES = len(CLASSES)

# # Re-create MobileNetV2 with the same classifier head you used
# model = models.mobilenet_v2(pretrained=False)
# model.classifier = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(model.last_channel, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, NUM_CLASSES)
# )

# # Load saved weights
# MODEL_PATH = 'best_model.pth'  # ← change to your actual path if different

# if not os.path.exists(MODEL_PATH):
#     print(f"ERROR: Model file not found: {MODEL_PATH}")
#     print("Please place 'best_model.pth' in the same folder as app.py or update MODEL_PATH")
# else:
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         print("✓ Underwater classification model loaded successfully")
#     except Exception as e:
#         print("Error loading model:", str(e))
#         print("Common causes: wrong number of classes, different architecture, or corrupted file")

# model = model.to(DEVICE)
# model.eval()

# # ─── Image preprocessing (must match training) ───────────────────
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ─── Prediction function ────────────────────────────────────────
# def predict_underwater_image(image_path):
#     if not os.path.exists(image_path):
#         return {"error": "Image file not found"}, None

#     try:
#         img = Image.open(image_path).convert('RGB')
#         img_tensor = transform(img).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             output = model(img_tensor)
#             probs = torch.softmax(output, dim=1)
#             confidence, pred_idx = torch.max(probs, 1)

#             pred_class = CLASSES[pred_idx.item()]
#             conf_pct = confidence.item() * 100

#         return {
#             "prediction": pred_class,
#             "confidence": round(conf_pct, 2),
#             "all_probs": {CLASSES[i]: round(p * 100, 2) for i, p in enumerate(probs[0].tolist())}
#         }, img

#     except Exception as e:
#         return {"error": str(e)}, None

# # ─── Main prediction route ──────────────────────────────────────
# @app.route('/prediction', methods=["GET", "POST"])
# def prediction():
#     if 'user_email' not in session:
#         return redirect(url_for('login'))

#     result = None
#     image_filename = None

#     if request.method == "POST":
#         file = request.files.get('file')
#         if file and file.filename != '':
#             filename = file.filename
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(image_path)
#             image_filename = filename

#             result, _ = predict_underwater_image(image_path)

          

#     return render_template('prediction.html',
#                            result=result,
#                            image_filename=image_filename)

# @app.route('/logout')
# def logout():
#     # Clear the session to log the user out
#     session.clear()
    
#     # Redirect to the login page or home page
#     return redirect(url_for('index'))  # Redirecting to login page after logout



# if __name__ == '__main__':
#     app.run(debug = True)

import mysql.connector, os, re
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter
import uuid
import matplotlib.pyplot as plt
import pymysql
from flask import Flask, session, render_template, request, redirect, url_for, send_from_directory
from timm import create_model
import cv2
from skimage import restoration, exposure, filters
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'admin'

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    port=3306,
    database='Under'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('register.html', message="Successfully Registered!")
            
            return render_template('register.html', message="This email ID already exists!")
        
        return render_template('register.html', message="Confirm password does not match!")
    
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = [i[0] for i in email_data]

        if email.upper() in email_data_list:
            query = "SELECT password, name FROM users WHERE email = %s"
            values = (email,)
            user_data = retrivequery1(query, values)
            
            if user_data:
                stored_password, name = user_data[0]
                
                if password == stored_password:
                    session['user_email'] = email
                    session['user_name'] = name
                    
                    return render_template('home.html', user_name=name)
                
                return render_template('login.html', message="Invalid Password!")
            
            return render_template('login.html', message="This email ID does not exist!")
        
        return render_template('login.html', message="This email ID does not exist!")
    
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    user_name = session.get('user_name')
    return render_template('home.html', user_name=user_name)

@app.route('/about')
def about():
    return render_template('about.html')

# Model performance reports (hardcoded from your provided results)
MODEL_REPORTS = {
    "mobilenet": {
        "classes": {
            "blue_tint": {"precision": 0.94, "recall": 0.97, "f1-score": 0.96, "support": 186},
            "blurry": {"precision": 0.99, "recall": 0.86, "f1-score": 0.92, "support": 174},
            "green_tint": {"precision": 0.88, "recall": 0.99, "f1-score": 0.93, "support": 160},
            "hazy": {"precision": 1.00, "recall": 0.99, "f1-score": 1.00, "support": 176},
            "high_contrast": {"precision": 0.87, "recall": 0.78, "f1-score": 0.82, "support": 175},
            "low_illumination": {"precision": 0.91, "recall": 0.98, "f1-score": 0.94, "support": 183},
            "noisy": {"precision": 1.00, "recall": 0.99, "f1-score": 1.00, "support": 182},
            "raw-890": {"precision": 0.77, "recall": 0.77, "f1-score": 0.77, "support": 184},
            "red_tint": {"precision": 0.98, "recall": 0.98, "f1-score": 0.98, "support": 185},
        },
        "accuracy": 0.93,
        "macro_avg": {"precision": 0.93, "recall": 0.93, "f1-score": 0.92},
        "weighted_avg": {"precision": 0.93, "recall": 0.93, "f1-score": 0.92}
    },
    "resnet": {
        "classes": {
            "blue_tint": {"precision": 0.98, "recall": 0.96, "f1-score": 0.97, "support": 167},
            "blurry": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92, "support": 185},
            "green_tint": {"precision": 0.94, "recall": 0.98, "f1-score": 0.96, "support": 192},
            "hazy": {"precision": 0.98, "recall": 1.00, "f1-score": 0.99, "support": 184},
            "high_contrast": {"precision": 0.88, "recall": 0.78, "f1-score": 0.83, "support": 174},
            "low_illumination": {"precision": 0.90, "recall": 0.98, "f1-score": 0.94, "support": 192},
            "noisy": {"precision": 0.99, "recall": 1.00, "f1-score": 1.00, "support": 160},
            "raw-890": {"precision": 0.75, "recall": 0.69, "f1-score": 0.72, "support": 172},
            "red_tint": {"precision": 0.96, "recall": 0.98, "f1-score": 0.97, "support": 179},
        },
        "accuracy": 0.92,
        "macro_avg": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92},
        "weighted_avg": {"precision": 0.92, "recall": 0.92, "f1-score": 0.92}
    },
    "cnn": {
        "classes": {
            "blue_tint": {"precision": 0.9532, "recall": 0.9645, "f1-score": 0.9588, "support": 169},
            "blurry": {"precision": 0.8901, "recall": 0.8571, "f1-score": 0.8733, "support": 189},
            "green_tint": {"precision": 0.9529, "recall": 1.0000, "f1-score": 0.9759, "support": 162},
            "hazy": {"precision": 0.9692, "recall": 1.0000, "f1-score": 0.9844, "support": 189},
            "high_contrast": {"precision": 0.9375, "recall": 0.7459, "f1-score": 0.8308, "support": 181},
            "low_illumination": {"precision": 0.9109, "recall": 0.9946, "f1-score": 0.9509, "support": 185},
            "noisy": {"precision": 1.0000, "recall": 0.9946, "f1-score": 0.9973, "support": 184},
            "raw-890": {"precision": 0.7513, "recall": 0.8256, "f1-score": 0.7867, "support": 172},
            "red_tint": {"precision": 0.9586, "recall": 0.9310, "f1-score": 0.9446, "support": 174},
        },
        "accuracy": 0.9234,
        "macro_avg": {"precision": 0.9249, "recall": 0.9237, "f1-score": 0.9225},
        "weighted_avg": {"precision": 0.9253, "recall": 0.9234, "f1-score": 0.9226}
    }
}

@app.route('/model', methods=['GET', 'POST'])
def model():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    selected_model = None
    report = None

    if request.method == 'POST':
        selected_model = request.form.get('modelSelect')
        if selected_model in MODEL_REPORTS:
            report = MODEL_REPORTS[selected_model]

    return render_template('model.html', 
                          selected_model=selected_model, 
                          report=report,
                          models_available=["mobilenet", "resnet", "cnn"])

# # Upload folders
# UPLOAD_FOLDER = 'static/uploads'
# ENHANCED_FOLDER = 'static/enhanced'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# # ────────────────────────────────────────────────────────────────
# #  MODEL SETUP - Underwater Image Degradation Classification
# # ────────────────────────────────────────────────────────────────

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLASSES = ['blue_tint', 'blurry', 'green_tint', 'hazy', 'high_contrast', 
#            'low_illumination', 'noisy', 'raw-890', 'red_tint']

# NUM_CLASSES = len(CLASSES)

# # Re-create MobileNetV2 with the same classifier head you used
# model = models.mobilenet_v2(pretrained=False)
# model.classifier = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(model.last_channel, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, NUM_CLASSES)
# )

# # Load saved weights
# MODEL_PATH = 'best_model.pth'

# if not os.path.exists(MODEL_PATH):
#     print(f"ERROR: Model file not found: {MODEL_PATH}")
#     print("Please place 'best_model.pth' in the same folder as app.py or update MODEL_PATH")
# else:
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         print("✓ Underwater classification model loaded successfully")
#     except Exception as e:
#         print("Error loading model:", str(e))

# model = model.to(DEVICE)
# model.eval()

# # ─── Image preprocessing ───────────────────────────────────────
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ────────────────────────────────────────────────────────────────
# #  ENHANCEMENT FUNCTIONS - Adaptive Underwater Image Enhancement
# # ────────────────────────────────────────────────────────────────

# def enhance_blue_tint(image):
#     """Correct blue/green color cast by balancing color channels"""
#     img = np.array(image).astype(np.float32) / 255.0
#     r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
#     # White balance using gray world assumption
#     r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
#     avg = (r_avg + g_avg + b_avg) / 3
    
#     r = np.clip(r * (avg / (r_avg + 1e-6)), 0, 1)
#     g = np.clip(g * (avg / (g_avg + 1e-6)), 0, 1)
#     b = np.clip(b * (avg / (b_avg + 1e-6)), 0, 1)
    
#     enhanced = np.stack([r, g, b], axis=2)
#     enhanced = (enhanced * 255).astype(np.uint8)
    
#     # Apply CLAHE for contrast enhancement
#     lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     return Image.fromarray(enhanced)

# def enhance_blurry(image):
#     """Sharpen blurry images using unsharp masking"""
#     img = np.array(image)
#     # Apply unsharp masking
#     gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
#     enhanced = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
#     # Additional sharpening
#     kernel = np.array([[-1,-1,-1],
#                        [-1, 9,-1],
#                        [-1,-1,-1]])
#     enhanced = cv2.filter2D(enhanced, -1, kernel)
    
#     return Image.fromarray(enhanced)

# def enhance_green_tint(image):
#     """Remove green cast typical in underwater images"""
#     img = np.array(image).astype(np.float32) / 255.0
    
#     # Reduce green channel
#     img[:,:,1] = img[:,:,1] * 0.7
    
#     # Enhance red and blue
#     img[:,:,0] = np.clip(img[:,:,0] * 1.3, 0, 1)  # Red
#     img[:,:,2] = np.clip(img[:,:,2] * 1.2, 0, 1)  # Blue
    
#     img = (img * 255).astype(np.uint8)
    
#     # Color correction
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     l = cv2.equalizeHist(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     return Image.fromarray(enhanced)

# def enhance_hazy(image):
#     """Dehaze using dark channel prior"""
#     img = np.array(image)
    
#     # Convert to float
#     img_float = img.astype(np.float64) / 255.0
    
#     # Estimate atmospheric light
#     dark_channel = np.min(img_float, axis=2)
#     top_percent = np.percentile(dark_channel, 95)
#     atmospheric_light = np.mean(img_float[dark_channel >= top_percent], axis=0)
    
#     # Estimate transmission map
#     transmission = 1 - 0.95 * dark_channel / np.mean(atmospheric_light)
#     transmission = np.clip(transmission, 0.1, 1)
    
#     # Recover scene radiance
#     enhanced = np.zeros_like(img_float)
#     for i in range(3):
#         enhanced[:,:,i] = (img_float[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
#     enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
#     # Apply contrast stretching
#     lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     return Image.fromarray(enhanced)

# def enhance_high_contrast(image):
#     """Reduce contrast and balance exposure"""
#     img = np.array(image)
    
#     # Convert to LAB and apply CLAHE with lower clip limit
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     # Slight gamma correction to reduce harshness
#     gamma = 0.9
#     enhanced = (np.power(enhanced.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)
    
#     return Image.fromarray(enhanced)

# def enhance_low_illumination(image):
#     """Brighten low-light images"""
#     img = np.array(image)
    
#     # Convert to LAB and apply CLAHE with high clip limit
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     # Gamma correction for brightness
#     gamma = 0.6
#     enhanced = (np.power(enhanced.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)
    
#     return Image.fromarray(enhanced)

# def enhance_noisy(image):
#     """Denoise using Non-Local Means"""
#     img = np.array(image)
    
#     # Apply Non-Local Means denoising
#     denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
#     # Slight sharpening after denoising
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     enhanced = cv2.filter2D(denoised, -1, kernel)
    
#     return Image.fromarray(enhanced)

# def enhance_raw_890(image):
#     """Generic enhancement for raw/degraded images"""
#     # Apply a combination of enhancement techniques
#     img = np.array(image)
    
#     # CLAHE for contrast
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     # Slight sharpening
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     enhanced = cv2.filter2D(enhanced, -1, kernel)
    
#     # Color balance
#     r, g, b = cv2.split(enhanced)
#     r = cv2.equalizeHist(r)
#     g = cv2.equalizeHist(g)
#     b = cv2.equalizeHist(b)
#     enhanced = cv2.merge([r, g, b])
    
#     return Image.fromarray(enhanced)

# def enhance_red_tint(image):
#     """Remove red tint and balance colors"""
#     img = np.array(image).astype(np.float32) / 255.0
    
#     # Reduce red channel
#     img[:,:,0] = img[:,:,0] * 0.8
    
#     # Enhance green and blue
#     img[:,:,1] = np.clip(img[:,:,1] * 1.2, 0, 1)  # Green
#     img[:,:,2] = np.clip(img[:,:,2] * 1.2, 0, 1)  # Blue
    
#     img = (img * 255).astype(np.uint8)
    
#     # Apply CLAHE
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     lab = cv2.merge([l, a, b])
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
#     return Image.fromarray(enhanced)

# # Enhancement mapping dictionary
# ENHANCEMENT_FUNCTIONS = {
#     'blue_tint': enhance_blue_tint,
#     'blurry': enhance_blurry,
#     'green_tint': enhance_green_tint,
#     'hazy': enhance_hazy,
#     'high_contrast': enhance_high_contrast,
#     'low_illumination': enhance_low_illumination,
#     'noisy': enhance_noisy,
#     'raw-890': enhance_raw_890,
#     'red_tint': enhance_red_tint
# }

# def enhance_image_by_class(image, predicted_class):
#     """Apply enhancement based on the predicted degradation class"""
#     if predicted_class in ENHANCEMENT_FUNCTIONS:
#         return ENHANCEMENT_FUNCTIONS[predicted_class](image)
#     else:
#         # Return original image if no enhancement function found
#         return image

# # ─── Prediction function ────────────────────────────────────────
# def predict_underwater_image(image_path):
#     if not os.path.exists(image_path):
#         return {"error": "Image file not found"}, None

#     try:
#         img = Image.open(image_path).convert('RGB')
#         img_tensor = transform(img).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             output = model(img_tensor)
#             probs = torch.softmax(output, dim=1)
#             confidence, pred_idx = torch.max(probs, 1)

#             pred_class = CLASSES[pred_idx.item()]
#             conf_pct = confidence.item() * 100

#         return {
#             "prediction": pred_class,
#             "confidence": round(conf_pct, 2),
#             "all_probs": {CLASSES[i]: round(p * 100, 2) for i, p in enumerate(probs[0].tolist())}
#         }, img

#     except Exception as e:
#         return {"error": str(e)}, None





# # ─── Main prediction and enhancement route ─────────────────────
# @app.route('/prediction', methods=["GET", "POST"])
# def prediction():
#     if 'user_email' not in session:
#         return redirect(url_for('login'))

#     result = None
#     image_filename = None
#     enhanced_filename = None
#     original_exists = False

#     if request.method == "POST":
#         file = request.files.get('file')
#         if file and file.filename != '':
#             # Save original image
#             filename = file.filename
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(image_path)
#             image_filename = filename

#             # Predict degradation
#             result, original_img = predict_underwater_image(image_path)
#             original_exists = True

#             # Apply enhancement if prediction was successful
#             if result and "error" not in result and original_img:
#                 predicted_class = result['prediction']
                
#                 # Enhance the image
#                 enhanced_img = enhance_image_by_class(original_img, predicted_class)
                
#                 # Save enhanced image
#                 enhanced_filename = f"enhanced_{filename}"
#                 enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
#                 enhanced_img.save(enhanced_path)
                
#                 # Add enhancement info to result
#                 result['enhancement_applied'] = True
#                 result['enhanced_filename'] = enhanced_filename

#     return render_template('prediction.html',
#                            result=result,
#                            image_filename=image_filename,
#                            enhanced_filename=enhanced_filename,
#                            original_exists=original_exists)

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('index'))




# Upload folder
UPLOAD_FOLDER = 'static/uploads'
ENHANCED_FOLDER = 'static/enhanced'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# ────────────────────────────────────────────────────────────────
#  MODEL SETUP - Underwater Image Degradation Classification
# ────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: These must match EXACTLY what you used during training
# (usually printed as dataset.classes or you can hardcode them)
CLASSES = ['blue_tint',
 'blurry',
 'green_tint',
 'hazy',
 'high_contrast',
 'low_illumination',
 'noisy',
 'raw-890',
 'red_tint']

NUM_CLASSES = len(CLASSES)

# Re-create MobileNetV2 with the same classifier head you used
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASSES)
)

# Load saved weights
MODEL_PATH = 'best_model.pth'  # ← change to your actual path if different

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    print("Please place 'best_model.pth' in the same folder as app.py or update MODEL_PATH")
else:
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✓ Underwater classification model loaded successfully")
    except Exception as e:
        print("Error loading model:", str(e))
        print("Common causes: wrong number of classes, different architecture, or corrupted file")

model = model.to(DEVICE)
model.eval()

# ─── Image preprocessing (must match training) ───────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ────────────────────────────────────────────────────────────────
#  ENHANCEMENT FUNCTIONS - Targeted per degradation class
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
#  ENHANCEMENT FUNCTIONS - AGGRESSIVE VERSION
# ────────────────────────────────────────────────────────────────

def apply_white_balance(image):
    """Manual Gray World White Balance implementation"""
    result = image.copy()
    result = result.astype(np.float32)
    
    # Calculate mean of each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    # Calculate gray value
    gray_value = (avg_b + avg_g + avg_r) / 3
    
    # Calculate gain for each channel (with safety checks)
    gain_b = gray_value / avg_b if avg_b > 0 else 1
    gain_g = gray_value / avg_g if avg_g > 0 else 1
    gain_r = gray_value / avg_r if avg_r > 0 else 1
    
    # Apply gains
    result[:, :, 0] *= gain_b
    result[:, :, 1] *= gain_g
    result[:, :, 2] *= gain_r
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_blurry(image):
    """AGGRESSIVE deblurring and sharpening"""
    result = image.copy()
    
    # 1. Strong unsharp masking
    gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
    result = cv2.addWeighted(result, 2.0, gaussian, -1.0, 0)
    
    # 2. Apply sharpening kernel
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    # 3. Apply another round of sharpening for severe blur
    kernel_sharpen2 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    result = cv2.filter2D(result, -1, kernel_sharpen2)
    
    # 4. Enhance contrast to make details pop
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    return result

def enhance_low_illumination(image):
    """AGGRESSIVE brightness enhancement for dark images"""
    result = image.copy()
    
    # 1. Convert to HSV
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. Aggressive gamma correction based on brightness
    mean_brightness = np.mean(v)
    if mean_brightness < 30:
        gamma = 0.3  # Very dark
    elif mean_brightness < 60:
        gamma = 0.4  # Dark
    elif mean_brightness < 90:
        gamma = 0.5  # Low light
    else:
        gamma = 0.7  # Slightly dark
    
    v = adjust_gamma(v, gamma)
    
    # 3. Boost brightness further
    v = cv2.add(v, 40)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    # 4. Boost saturation for better colors
    s = cv2.multiply(s, 1.5)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    # 5. Apply CLAHE for contrast
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    return result

def enhance_noisy(image):
    """AGGRESSIVE noise removal while preserving details"""
    result = image.copy()
    
    # 1. Strong bilateral filtering
    result = cv2.bilateralFilter(result, 15, 100, 100)
    
    # 2. Apply median blur for salt-pepper noise
    result = cv2.medianBlur(result, 5)
    
    # 3. Apply Gaussian blur
    result = cv2.GaussianBlur(result, (5, 5), 1.0)
    
    # 4. Apply Non-local Means Denoising (strong)
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    
    # 5. Slight sharpening to recover details lost in denoising
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    return result

def enhance_color_cast(image, tint_type):
    """AGGRESSIVE color correction"""
    result = image.copy()
    
    # 1. Strong white balance
    result = apply_white_balance(result)
    
    # 2. Additional channel-specific adjustments
    b, g, r = cv2.split(result)
    
    if 'blue' in tint_type:
        # Reduce blue, boost red/green
        b = cv2.multiply(b, 0.7)
        g = cv2.multiply(g, 1.2)
        r = cv2.multiply(r, 1.3)
    elif 'green' in tint_type:
        # Reduce green, boost red/blue
        b = cv2.multiply(b, 1.2)
        g = cv2.multiply(g, 0.7)
        r = cv2.multiply(r, 1.2)
    elif 'red' in tint_type:
        # Reduce red, boost blue/green
        b = cv2.multiply(b, 1.3)
        g = cv2.multiply(g, 1.2)
        r = cv2.multiply(r, 0.7)
    
    b = np.clip(b, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)
    
    result = cv2.merge([b, g, r])
    
    # 3. Aggressive CLAHE
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    return result

def enhance_hazy(image):
    """AGGRESSIVE haze removal"""
    result = image.copy()
    
    # 1. Convert to YUV and aggressive histogram equalization
    img_yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 2. Apply CLAHE
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # 3. Strong sharpening
    kernel_sharpen = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    # 4. Increase saturation
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.3)
    s = np.clip(s, 0, 255).astype(np.uint8)
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    return result

def enhance_high_contrast(image):
    """Reduce harsh contrast and balance exposure"""
    result = image.copy()
    
    # 1. Convert to LAB
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. CLAHE with lower clip limit for contrast reduction
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 3. Apply mild gamma to balance
    l = adjust_gamma(l, 0.9)
    
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # 4. Slight shadow highlight adjustment
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Compress highlights, lift shadows
    v = np.clip(v, 20, 235)
    v = cv2.normalize(v, None, 30, 225, cv2.NORM_MINMAX)
    
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    return result

def enhance_raw_890(image):
    """COMPREHENSIVE enhancement for raw images"""
    result = image.copy()
    
    # 1. White balance
    result = apply_white_balance(result)
    
    # 2. CLAHE for contrast
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    # 4. Saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)
    s = np.clip(s, 0, 255).astype(np.uint8)
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    # 5. Gamma correction for overall brightness
    result = adjust_gamma(result, 0.8)
    
    return result

# Keep the same enhance_image dispatcher function
def enhance_image(image, prediction_class):
    """Main enhancement dispatcher function"""
    
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        image = cv2.imread(image)
    
    # Dispatch to appropriate enhancement function
    if prediction_class in ['blue_tint', 'green_tint']:
        enhanced = enhance_color_cast(image, prediction_class)
    elif prediction_class == 'blurry':
        enhanced = enhance_blurry(image)
    elif prediction_class == 'hazy':
        enhanced = enhance_hazy(image)
    elif prediction_class == 'high_contrast':
        enhanced = enhance_high_contrast(image)
    elif prediction_class == 'low_illumination':
        enhanced = enhance_low_illumination(image)
    elif prediction_class == 'noisy':
        enhanced = enhance_noisy(image)
    elif prediction_class == 'raw-890':
        enhanced = enhance_raw_890(image)
    elif prediction_class == 'red_tint':
        enhanced = enhance_color_cast(image, 'red_tint')
    else:
        enhanced = enhance_raw_890(image)
    
    return enhanced

# ─── Prediction function ────────────────────────────────────────
def predict_underwater_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "Image file not found"}, None

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

            pred_class = CLASSES[pred_idx.item()]
            conf_pct = confidence.item() * 100

            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probs, 3)
            top_predictions = [
                {
                    "class": CLASSES[idx.item()],
                    "confidence": round(prob.item() * 100, 2)
                }
                for prob, idx in zip(top_probs[0], top_indices[0])
            ]

        return {
            "prediction": pred_class,
            "confidence": round(conf_pct, 2),
            "top_predictions": top_predictions,
            "all_probs": {CLASSES[i]: round(p * 100, 2) for i, p in enumerate(probs[0].tolist())}
        }, img

    except Exception as e:
        return {"error": str(e)}, None
from datetime import datetime
from werkzeug.utils import secure_filename  # ← 
# ─── Main prediction route ──────────────────────────────────────
@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    result = None
    image_filename = None
    enhanced_filename = None

    if request.method == "POST":
        file = request.files.get('file')
        if file and file.filename != '':
            # Save original image
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Original image
            original_filename = f"{name}_{timestamp}{ext}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(image_path)
            image_filename = original_filename

            # Get prediction
            result, pil_img = predict_underwater_image(image_path)
            
            if "error" not in result:
                # Apply enhancement
                enhanced_img = enhance_image(image_path, result["prediction"])
                
                # Save enhanced image
                enhanced_filename = f"{name}_{timestamp}_enhanced{ext}"
                enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
                cv2.imwrite(enhanced_path, enhanced_img)
                
                # Add to result
                result["enhanced_filename"] = enhanced_filename

    return render_template('prediction.html',
                           result=result,
                           image_filename=image_filename,
                           enhanced_filename=enhanced_filename if 'enhanced_filename' in locals() else None)

@app.route('/logout')
def logout():
    # Clear the session to log the user out
    session.clear()
    
    # Redirect to the login page or home page
    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)