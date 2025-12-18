
from flask import Flask, render_template, request, jsonify, Response
import pickle
import joblib
import pandas as pd
import numpy as np
import os
import random
from video_processor import TrafficCamera

app = Flask(__name__)

# Load Model logic
MODEL_PATH = r"c:\Users\zadkiel\Downloads\traffic_reducer_dataset-20251211T152732Z-3-001\traffic_reducer_dataset\modelo_entrenado\modelo_semaforo_ia.pkl"

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except:
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

model = load_model()

# Initialize Camera
YOUTUBE_URL = "https://www.youtube.com/watch?v=ByED80IKdIU" # User's video
camera = TrafficCamera(YOUTUBE_URL)
camera.start()

@app.route('/')
def home():
    return render_template('index.html')

def generate_frames():
    while True:
        frame_bytes = camera.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    try:
        # Check if we should use Real Camera Data
        # If frontend sends specific flag or just always use camera in this mode?
        # Let's check a "mode" data param
        use_live = data.get('live_mode', False)
        
        if use_live:
            counts = camera.get_counts()
            norte = float(counts['norte'])
            sur = float(counts['sur'])
            este = float(counts['este'])
            oeste = float(counts['oeste'])
        else:
            norte = float(data.get('norte', 0))
            sur = float(data.get('sur', 0))
            este = float(data.get('este', 0))
            oeste = float(data.get('oeste', 0))
        
        # Prepare input for model (Original AI)
        input_data = [[norte, sur, este, oeste]]
        
        # LOGIC OVERRIDE: User wants priority based on HIGHEST COUNT
        # "que se cambie segun el numero de carros que mayor tenga"
        traffic_values = [norte, sur, este, oeste]
        
        # If total traffic is very low, default to 0 (Norte) or keep previous? 
        # For responsiveness, just pick max.
        if sum(traffic_values) > 0:
            result = int(np.argmax(traffic_values))
        else:
            result = 0 # Default if empty

        # Update Camera Status for OSD
        if use_live:
            camera.set_phase(result)
        
        return jsonify({
            'prediction': result,
            'traffic_data': {
                'norte': norte, 'sur': sur, 'este': este, 'oeste': oeste
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/simulate', methods=['GET'])
def simulate():
    # Generate random traffic data for live simulation
    data = {
        'norte': random.randint(0, 80),
        'sur': random.randint(0, 80),
        'este': random.randint(0, 80),
        'oeste': random.randint(0, 80)
    }
    return jsonify(data)

if __name__ == '__main__':
    print("Starting Traffic AI Server...")
    print("Go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
