# =======================================================================
# Pest Detection Code and API Report Sending
# =======================================================================
import os
import time
import numpy as np
import cv2
import requests # HTTP requests library
from picamera2 import Picamera2
import tensorflow as tf

# =================== SETTINGS ===================
MODEL_DIR = "plant_disease_tomato_tf"      # SavedModel folder
LABELS_FILE = "labels.txt"                 # Labels file
IMG_SIZE = (150, 150)
CONF_TH = 0.85 # Increased confidence level to reduce false alerts

# ----------- Server Settings (MUST BE UPDATED) -----------
# If the server is running locally on 192.168.1.X
API_URL = "http://192.168.137.10:8000" 
PEST_REPORT_ENDPOINT = f"{API_URL}/data/pest-report/"
PLANT_NAME = "Tomato" # Can be modified

# ---------------- MODE ----------------
LIVE_MODE = True
TEST_IMAGE = "test_tomato.jpg"

# ---------------- Professional Recommendations (Pest Database) ----------------
# The key must match the output label from the AI model
PEST_RECOMMENDATIONS = {
    "Tomato___Bacterial_spot": {
        "pest_name": "Bacterial Spot",
        "recommendation": "Use copper-based biopesticides, avoid overhead irrigation, and remove infected leaves to limit spread."
    },
    "Tomato___Late_blight": {
        "pest_name": "Late Blight",
        "recommendation": "Immediately spray systemic fungicides, ensure good plant ventilation, and monitor humidity levels."
    },
    "Tomato___White_rot": {
        "pest_name": "White Rot",
        "recommendation": "Completely remove infected plants. Use soil fungicides, and ensure pruning tools are sterilized."
    },
    "Tomato___healthy": {
        "pest_name": "Healthy",
        "recommendation": "The plant is healthy. Continue monitoring and irrigation as needed."
    },
    # Add other pests here...
}


# =================== Load Labels and Model ===================
# (Code remains largely the same, focusing on translations)
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    LABELS = [l.strip() for l in f if l.strip()]

print(f"[✓] Loaded {len(LABELS)} labels.")

print(f"\n[ ] Loading SavedModel from: {MODEL_DIR} ...")
saved_model = tf.saved_model.load(MODEL_DIR)
infer = saved_model.signatures["serving_default"]
print("[✓] SavedModel loaded successfully!\n")

def predict_image(img):
    # Prediction function
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    inp = tf.convert_to_tensor(img)
    outputs = infer(inp)
    probs = list(outputs.values())[0].numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return LABELS[idx], conf


# =================== API Report Sending Function ===================
def send_pest_report(pred_key, confidence):
    report_config = PEST_RECOMMENDATIONS.get(pred_key, PEST_RECOMMENDATIONS["Tomato___healthy"])
    
    if pred_key == "Tomato___healthy":
        # We don't send healthy reports unless a full log is desired
        print("[ ] Health check sent.")
        return
        
    report_data = {
        "pest_name": report_config["pest_name"],
        "plant_name": PLANT_NAME,
        "detection_certainty": confidence,
        "recommendation": report_config["recommendation"]
    }
        
    try:
        response = requests.post(PEST_REPORT_ENDPOINT, json=report_data, timeout=5)
        
        if response.status_code == 201:
            print(f"[✓] Report ({report_data['pest_name']}) submitted to API.")
        else:
            print(f"✗ API Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Network Error: Could not connect to API at {API_URL}. Is the backend running? {e}")


# =================== Direct Execution ===================
if LIVE_MODE is False:
    # Test Mode
    # ... (Test logic skipped for brevity)
    pass
else:
    # Live Camera Mode
    print("[•] LIVE CAMERA MODE")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640,480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            pred, conf = predict_image(frame)

            if conf >= CONF_TH:
                if "healthy" not in pred.lower():
                    print(f"!!! Pest Detected: {pred} ({conf*100:.1f}%)")
                    send_pest_report(pred, conf)
                else:
                    print(f"[~] System Health Check: {pred} ({conf*100:.1f}%)")
            else:
                print(f"<?> Low confidence ({conf*100:.1f}%), ignoring...")

            time.sleep(10) # Check every 10 seconds

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        picam2.stop()
