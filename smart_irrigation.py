# ============================================================
# Smart Irrigation System (AI Model & Manual Control)
# ============================================================

# ----------- Import Libraries -----------
import warnings
warnings.filterwarnings("ignore") 

import time, math, statistics, csv, os
from collections import deque
from datetime import datetime, timezone
import numpy as np
import joblib
import RPi.GPIO as GPIO
import spidev, board, adafruit_dht
import requests # Library for sending and receiving API data

# ----------- User Settings -----------
MODEL_PATH = "models/irrigation_model_merged.pkl"
RELAY_PIN = 17
ACTIVE_HIGH = True 
DRY_RUN = False 
SOIL_CH = 0
WET = 233
DRY = 619
THRESH_OVERRIDE = None
EMERGENCY_ON_PCT = 20.0 
BURST_ON_SEC = 4
REST_SEC = 5
MIN_ON_SEC = 6
MIN_OFF_SEC = 3
MAX_ON_SEC = 60
MAX_MIN_PER_HOUR = 8
HOURLY_BUCKET = 3600

# ----------- Server Settings (MUST BE UPDATED) -----------
# Ensure this is the full HTTPS public URL from Render
API_URL = "https://backend-qkbr.onrender.com" 
SENSOR_DATA_ENDPOINT = f"{API_URL}/data/sensor/"
CONTROL_STATUS_ENDPOINT = f"{API_URL}/control/status/"
REPORT_INTERVAL = 5 # Send data every 5 seconds
CONTROL_CHECK_INTERVAL = 3 # Check manual control status every 3 seconds
API_TIMEOUT = 15 # Increased timeout to 15s to handle weak network connections

# ----------- Load AI Model -----------
# Ensure the model file is present in the specified path
try:
    bundle = joblib.load(MODEL_PATH)
    MODEL = bundle["model"]
    FEATURES = bundle["features"]
    THRESH = bundle.get("threshold", 0.06)
except FileNotFoundError:
    print(f"ERROR: AI Model not found at {MODEL_PATH}. Running in fail-safe mode (using thresholds).")
    MODEL = None
    THRESH = 0.0 # Use 0.0 threshold to default to the Emergency/Manual modes only.

if THRESH_OVERRIDE is not None:
    THRESH = float(THRESH_OVERRIDE)


# ----------- Hardware Setup (GPIO, SPI, DHT22) -----------
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW if ACTIVE_HIGH else GPIO.HIGH)

def relay_set(on: bool):
    """Turn the pump ON/OFF"""
    if DRY_RUN:
        return
    if ACTIVE_HIGH:
        GPIO.output(RELAY_PIN, GPIO.HIGH if on else GPIO.LOW)
    else:
        GPIO.output(RELAY_PIN, GPIO.LOW if on else GPIO.HIGH)

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 135000
spi.mode = 0

def read_adc(ch=0):
    r = spi.xfer2([1,(8+ch)<<4,0])
    return ((r[1]&3)<<8)|r[2]

def adc_to_pct(v, wet=WET, dry=DRY):
    v = max(min(v, dry), wet)
    return round(100.0*(dry - v)/(dry - wet), 1)

dht = adafruit_dht.DHT22(board.D4)

def read_dht_safe():
    try:
        t = dht.temperature
        h = dht.humidity
        if (t is not None) and (h is not None):
            return float(t), float(h)
    except Exception:
        pass
    return None, None

def vpd_kpa(temp_c, rh):
    """Calculate Vapor Pressure Deficit (VPD)"""
    if (temp_c is None) or (rh is None): return None
    es = 0.6108 * math.exp((17.27*temp_c)/(temp_c+237.3))
    ea = es * (rh/100.0)
    return es - ea


# ----------- API Functions -----------

def send_sensor_data(temp, hum, soil_pct, irrigation_status, ai_reason):
    """Send sensor data and decision to the server."""
    temp_safe = temp if temp is not None else 25.0
    hum_safe = hum if hum is not None else 50.0
    ai_decision_text = f"Irrigation required (Reason: {ai_reason})" if irrigation_status else "Irrigation not required"
    
    payload = {
        "temperature": temp_safe,
        "humidity": hum_safe,
        "soil_moisture": soil_pct,
        "irrigation_status": irrigation_status,
        "ai_decision": ai_decision_text
    }
    
    try:
        # Using the increased timeout for better network resilience
        requests.post(SENSOR_DATA_ENDPOINT, json=payload, timeout=API_TIMEOUT) 
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Network Error sending data. Timeout: {API_TIMEOUT}s")

def get_manual_control_status():
    """Fetch manual control status from the server."""
    try:
        # Using the increased timeout
        response = requests.get(CONTROL_STATUS_ENDPOINT, timeout=API_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get('manual_enabled', False), data.get('pump_command', False)
        # Handle 404/400 errors safely
        return False, False 
    except requests.exceptions.RequestException:
        # Revert to safe automated mode on connection failure
        return False, False 


# ----------- Filters and Buffers -----------
MEDIAN_N = 9
AVG_WINDOW = 12
buf = deque(maxlen=AVG_WINDOW)
last_soil = None
last30 = deque(maxlen=30)
# State to help monitor manual mode activity
last_manual_enabled = False 

# ----------- Status Variables -----------
pump_on = False
last_change = time.time()
on_start = 0.0
rest_until = 0.0
hour_window_start = time.time()
run_sec_this_hour = 0
last_report_time = time.time()
last_control_check = time.time()


# ----------- Logging Setup -----------
logfile = "ai_irrigation_log.csv"
print(f"Smart AI Irrigation Started | THRESH={THRESH:.3f} | ACTIVE_HIGH={ACTIVE_HIGH} | DRY_RUN={DRY_RUN} | API_TIMEOUT={API_TIMEOUT}s\n")
with open(logfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp","temp_C","hum_%","vpd","adc_raw","soil_%","soil_ma","delta_soil","proba","decision","reason","pump_on"])

    try:
        while True:
            now = time.time()
            global last_manual_enabled

            # Update hourly monitoring window
            if now - hour_window_start >= HOURLY_BUCKET:
                hour_window_start = now
                run_sec_this_hour = 0

            # ---------------- 1. Readings and Processing ----------------
            vals = [read_adc(SOIL_CH) for _ in range(MEDIAN_N)]
            med = int(statistics.median(vals))
            buf.append(med)
            adc_smooth = sum(buf)//len(buf)
            soil = adc_to_pct(adc_smooth)

            temp, hum = read_dht_safe()
            vpd = vpd_kpa(temp, hum)
            hour = int(datetime.now(timezone.utc).strftime("%H"))
            sin_h = math.sin(2*math.pi*hour/24.0)
            cos_h = math.cos(2*math.pi*hour/24.0)

            last30.append(soil)
            soil_ma = sum(last30)/len(last30)
            delta = 0.0 if last_soil is None else soil - last_soil
            last_soil = soil
            
            # ---------------- 2. Manual Override Check ----------------
            manual_enabled = False
            pump_command_manual = False
            
            # Check manual control status
            if now - last_control_check >= CONTROL_CHECK_INTERVAL:
                manual_enabled, pump_command_manual = get_manual_control_status()
                last_control_check = now
            
            # Log any change in manual mode state
            if manual_enabled != last_manual_enabled:
                print(f"\n*** SYSTEM STATE CHANGE: Manual Mode {'ENABLED' if manual_enabled else 'DISABLED'} ***\n")
                last_manual_enabled = manual_enabled
            
            # ---------------- 3. AI Decision (Only if not Manual) ----------------
            final_decision = False
            reason = "NO"

            if MODEL:
                # Prepare AI inputs
                row = {
                    "temperature_C": temp if temp is not None else 25.0,
                    "humidity_air_%": hum if hum is not None else 50.0,
                    "soil_moisture_%": soil,
                    "hour": hour,
                    "sin_hour": sin_h,
                    "cos_hour": cos_h,
                    "soil_moisture_ma": soil_ma,
                    "delta_soil": delta,
                    "vpd_kPa": vpd if vpd is not None else 1.0,
                }
                X = np.array([[row.get(f, 0.0) for f in FEATURES]], dtype=float)
                proba = float(MODEL.predict_proba(X)[0,1])
            else:
                proba = 0.0 # If model loading failed, assume low probability

            decision_ai = (proba >= THRESH)
            decision_emg = (soil <= EMERGENCY_ON_PCT)
            
            # ---------------- 4. Final Decision Determination ----------------
            if manual_enabled:
                # OVERRIDE: Execute user's command
                final_decision = pump_command_manual
                reason = "MANUAL"
            else:
                # AUTOMATIC: Execute AI or Emergency rule
                final_decision = decision_ai or decision_emg
                reason = "AI" if decision_ai else ("EMERGENCY" if decision_emg else "NO")

            in_rest = now < rest_until
            
            # ---------------- 5. Pump Control Execution ----------------
            
            # Turn ON irrigation
            if (not pump_on) and final_decision and (not in_rest) and (now-last_change)>=MIN_OFF_SEC:
                if (run_sec_this_hour/60.0) < MAX_MIN_PER_HOUR:
                    relay_set(True)
                    pump_on=True
                    on_start=now
                    last_change=now

            # Turn OFF after duration or due to manual OFF command
            is_safety_cut_off = pump_on and (now - on_start) >= MAX_ON_SEC
            is_pulse_end = pump_on and (now - on_start) >= BURST_ON_SEC
            is_manual_off = manual_enabled and pump_on and not pump_command_manual

            if (is_safety_cut_off or is_pulse_end or is_manual_off) and (now - last_change) >= MIN_ON_SEC:
                relay_set(False)
                pump_on=False
                last_change=now
                rest_until = now + REST_SEC
                if is_safety_cut_off:
                    print("Safety cutoff — forced OFF")
                elif is_manual_off:
                    print("Manual Command OFF received.")
                else:
                    print("Pump OFF (pulse ended / soak period)")


            if pump_on:
                run_sec_this_hour = min(HOURLY_BUCKET, run_sec_this_hour + 1)

            # ---------------- 6. API Data Sending ----------------
            if now - last_report_time >= REPORT_INTERVAL:
                send_sensor_data(temp, hum, soil, pump_on, reason)
                last_report_time = now

            # ---------------- 7. Logging and Printing ----------------
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                temp, hum, round(vpd,3) if vpd is not None else "",
                med, round(soil,1), round(soil_ma,1), round(delta,2),
                round(proba,3), int(final_decision), reason, int(pump_on)
            ])
            f.flush()

            print(f"{datetime.now().strftime('%H:%M:%S')} | T:{temp if temp else 'NA'}°C H:{hum if hum else 'NA'}% "
                  f"Soil:{soil:.1f}% p={proba:.3f} | Decision:{'ON' if pump_on else 'OFF'} ({reason}) | Manual:{'YES' if manual_enabled else 'NO'}")

            time.sleep(1.5)

    except KeyboardInterrupt:
        pass
    finally:
        relay_set(False)
        GPIO.cleanup()
        spi.close()
        print("\nSystem stopped safely by the user.\n")
