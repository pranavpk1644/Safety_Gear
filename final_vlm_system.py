import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# --- CONFIGURATION ---
DANGER_ZONE = {"x_min": 200, "x_max": 400, "y_min": 200, "y_max": 400}

# 1. LOAD YOLO (Fast detection)
print("1/2: Loading YOLO (Eyes)...")
yolo_model = YOLO('yolov8n.pt') 

# 2. LOAD VLM (Brain)
print("2/2: Loading BLIP VLM (Brain)... This might take a minute...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vlm_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print("✅ Systems Ready.")

try:
    h_matrix = np.load("calibration_matrix.npy")
except:
    print("❌ ERROR: Run setup_calibration.py first.")
    exit()

# --- THE VLM FUNCTION ---
def ask_vlm(frame, box, question="Is the person wearing a high visibility vest?"):
    """
    Crops the person and asks the AI a question.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Crop the person from the image
    person_img_cv = frame[y1:y2, x1:x2]
    if person_img_cv.size == 0: return False
    
    # Convert OpenCV (BGR) to AI Format (RGB)
    person_img_pil = Image.fromarray(cv2.cvtColor(person_img_cv, cv2.COLOR_BGR2RGB))
    
    # Prepare inputs for AI
    inputs = processor(person_img_pil, question, return_tensors="pt")
    
    # AI Thinks (Inference)
    out = vlm_model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"DEBUG: VLM Answered -> '{answer}'") # Look at terminal to see what it thinks
    return answer.lower()

# --- DASHBOARD SETUP ---
map_size = 600
base_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
# Grid
for i in range(0, map_size, 50):
    cv2.line(base_map, (i, 0), (i, map_size), (50, 50, 50), 1)
    cv2.line(base_map, (0, i), (map_size, i), (50, 50, 50), 1)
# Zone
overlay = base_map.copy()
cv2.rectangle(overlay, (DANGER_ZONE["x_min"], DANGER_ZONE["y_min"]), 
              (DANGER_ZONE["x_max"], DANGER_ZONE["y_max"]), (0, 0, 255), -1)
cv2.addWeighted(overlay, 0.3, base_map, 0.7, 0, base_map)

alert_log = [] 

def draw_ui(frame, map_img):
    h1, w1 = frame.shape[:2]
    h2, w2 = map_img.shape[:2]
    canvas_h = max(h1, h2) + 150
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[0:h1, 0:w1] = frame
    canvas[0:h2, w1:w1+w2] = map_img
    
    cv2.putText(canvas, "SYSTEM LOG:", (20, h1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, msg in enumerate(alert_log[-3:]):
        color = (0, 0, 255) if "VIOLATION" in msg else (0, 255, 0)
        cv2.putText(canvas, f"> {msg}", (20, h1 + 60 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return canvas

cap = cv2.VideoCapture(0)
frame_count = 0
last_scan_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (600, 600))
    current_map = base_map.copy()
    
    results = yolo_model(frame, stream=True, classes=0, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Map Position
            feet_x, feet_y = (x1 + x2) / 2, y2
            pts = np.array([[[feet_x, feet_y]]], dtype=float)
            trans_pt = cv2.perspectiveTransform(pts, h_matrix)
            mx, my = int(trans_pt[0][0][0]), int(trans_pt[0][0][1])
            
            # Check Zone
            in_danger = (DANGER_ZONE["x_min"] < mx < DANGER_ZONE["x_max"]) and \
                        (DANGER_ZONE["y_min"] < my < DANGER_ZONE["y_max"])
            
            status = "SAFE"
            box_color = (0, 255, 0)
            
            if in_danger:
                # INTELLIGENT SCANNING (Only scan once every 3 seconds to save lag)
                current_time = datetime.now().timestamp()
                
                if current_time - last_scan_time > 3.0: 
                    # TRIGGER VLM
                    status = "SCANNING..."
                    cv2.putText(frame, "AI ANALYZING...", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow("Industrial Safety System", draw_ui(frame, current_map)) # Force update screen
                    cv2.waitKey(1) # Allow screen to draw
                    
                    # Ask the VLM
                    answer = ask_vlm(frame, [x1, y1, x2, y2], "Is the person wearing a high visibility vest?")
                    
                    # Logic
                    if "yes" in answer:
                        last_status = "VERIFIED"
                        log_msg = "PPE VERIFIED (AI CONFIRMED)"
                    else:
                        last_status = "VIOLATION"
                        log_msg = "VIOLATION: NO VEST DETECTED"
                        
                    alert_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_msg}")
                    last_scan_time = current_time
                
                # Show Last Known Status
                if "VERIFIED" in str(alert_log): 
                    box_color = (0, 255, 255)
                    status = "PPE VERIFIED"
                else:
                    box_color = (0, 0, 255)
                    status = "VIOLATION"

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.circle(current_map, (mx, my), 10, (255, 255, 0), -1)

    cv2.imshow("Industrial Safety System", draw_ui(frame, current_map))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()