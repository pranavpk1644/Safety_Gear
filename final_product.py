import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# --- CONFIGURATION ---
DANGER_ZONE = {"x_min": 200, "x_max": 400, "y_min": 200, "y_max": 400}

# Load Models
print("Initializing Security System...")
model = YOLO('yolov8n.pt') 

# Load Calibration
try:
    h_matrix = np.load("calibration_matrix.npy")
except:
    print("❌ ERROR: Missing 'calibration_matrix.npy'. Run setup_calibration.py first.")
    exit()

# --- GENERATE DARK MODE MAP ---
# We create a technical looking grid instead of plain white
map_size = 600
base_map = np.zeros((map_size, map_size, 3), dtype=np.uint8) # Black background

# Draw Grid Lines
for i in range(0, map_size, 50):
    cv2.line(base_map, (i, 0), (i, map_size), (50, 50, 50), 1)
    cv2.line(base_map, (0, i), (map_size, i), (50, 50, 50), 1)

# Draw Danger Zone (Semi-transparent Red)
overlay = base_map.copy()
cv2.rectangle(overlay, (DANGER_ZONE["x_min"], DANGER_ZONE["y_min"]), 
              (DANGER_ZONE["x_max"], DANGER_ZONE["y_max"]), (0, 0, 255), -1)
cv2.addWeighted(overlay, 0.3, base_map, 0.7, 0, base_map) # Blend it
cv2.rectangle(base_map, (DANGER_ZONE["x_min"], DANGER_ZONE["y_min"]), 
              (DANGER_ZONE["x_max"], DANGER_ZONE["y_max"]), (0, 0, 255), 2) # Border

# --- STATE VARIABLES ---
scan_timer = 0
alert_log = [] # Stores messages like ["10:00 - Entered Zone"]

def draw_ui(frame, map_img):
    # Combine them
    h1, w1 = frame.shape[:2]
    h2, w2 = map_img.shape[:2]
    
    # Create a black canvas to hold both + the log
    canvas_h = max(h1, h2) + 150 # Extra space at bottom for logs
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place images
    canvas[0:h1, 0:w1] = frame
    canvas[0:h2, w1:w1+w2] = map_img
    
    # Draw Log Section
    cv2.putText(canvas, "SYSTEM LOG:", (20, h1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, msg in enumerate(alert_log[-3:]): # Show last 3 messages
        cv2.putText(canvas, f"> {msg}", (20, h1 + 60 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
    return canvas

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize frame to match map height (makes UI cleaner)
    frame = cv2.resize(frame, (600, 600))
    current_map = base_map.copy()
    
    # Detection
    results = model(frame, stream=True, classes=0, verbose=False)
    
    person_detected = False
    
    for r in results:
        for box in r.boxes:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Feet Logic
            feet_x, feet_y = (x1 + x2) / 2, y2
            
            # Map Logic
            pts = np.array([[[feet_x, feet_y]]], dtype=float)
            trans_pt = cv2.perspectiveTransform(pts, h_matrix)
            mx, my = int(trans_pt[0][0][0]), int(trans_pt[0][0][1])
            
            # Danger Check
            in_danger = (DANGER_ZONE["x_min"] < mx < DANGER_ZONE["x_max"]) and \
                        (DANGER_ZONE["y_min"] < my < DANGER_ZONE["y_max"])
            
            # Visuals
            color = (0, 255, 0)
            status = "SAFE"
            
            if in_danger:
                color = (0, 0, 255)
                # Scanning Logic
                scan_timer += 1
                
                if scan_timer < 30: # First 1 second (approx)
                    status = "SCANNING PPE..."
                    cv2.rectangle(frame, (x1, y1), (x2, y1+20), (255, 0, 0), -1) # Blue progress bar
                else:
                    status = "ALERT: NO PPE"
                    if scan_timer == 31: # Trigger log once
                        ts = datetime.now().strftime("%H:%M:%S")
                        alert_log.append(f"{ts} - VIOLATION DETECTED")
                
            else:
                scan_timer = 0 # Reset if they walk out
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Map Dot
            cv2.circle(current_map, (mx, my), 10, (255, 255, 0), -1) # Cyan dot
            cv2.line(current_map, (mx, my), (mx, my-20), (255, 255, 255), 1) # Direction indicator

    # Final Dashboard
    final_ui = draw_ui(frame, current_map)
    cv2.imshow("Industrial Safety Digital Twin", final_ui)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()