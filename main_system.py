import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
# These match the Red Square we drew in 'setup_calibration.py'
DANGER_ZONE_X_MIN = 200
DANGER_ZONE_X_MAX = 400
DANGER_ZONE_Y_MIN = 200
DANGER_ZONE_Y_MAX = 400

# Load the AI Model (YOLOv8 Nano - fast and light)
print("Loading AI Model... (this takes a few seconds)")
model = YOLO('yolov8n.pt') 

# Load your Calibration Data
try:
    h_matrix = np.load("calibration_matrix.npy")
    map_template = cv2.imread("map_image.jpg")
    map_template = cv2.resize(map_template, (600, 600)) # Ensure size matches
    print("✅ System Loaded Successfully")
except Exception as e:
    print("❌ ERROR: Could not load calibration files. Did you run setup_calibration.py?")
    exit()

# Start Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Create a fresh map for this frame (so dots don't smear)
    map_display = map_template.copy()

    # 1. AI DETECTION
    results = model(frame, stream=True, classes=0, verbose=False) # class 0 = Person

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the box around the person
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 2. FIND FEET (Bottom Center of the box)
            feet_x = (x1 + x2) / 2
            feet_y = y2 
            
            # 3. MATHEMATICAL MAPPING (The "Digital Twin" Magic)
            # We warp the feet coordinates using your calibration matrix
            point_vector = np.array([[[feet_x, feet_y]]], dtype=float)
            transformed_point = cv2.perspectiveTransform(point_vector, h_matrix)
            
            # These are your coordinates on the 2D Map
            map_x = int(transformed_point[0][0][0])
            map_y = int(transformed_point[0][0][1])
            
            print(f"I see you at: X={map_x}, Y={map_y}")

            # 4. SAFETY LOGIC CHECK
            # Are the feet inside the Red Zone coordinates?
            in_danger = (DANGER_ZONE_X_MIN < map_x < DANGER_ZONE_X_MAX) and \
                        (DANGER_ZONE_Y_MIN < map_y < DANGER_ZONE_Y_MAX)

            # 5. VISUALIZATION
            if in_danger:
                color = (0, 0, 255) # Red
                label = "DANGER ZONE"
                status_text = "STATUS: UNSAFE"
            else:
                color = (0, 255, 0) # Green
                label = "Safe"
                status_text = "STATUS: SAFE"
            
            # Draw on Camera Feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw on Digital Map (The Blue Dot is You)
            cv2.circle(map_display, (map_x, map_y), 15, (255, 0, 0), -1) # Blue dot
            cv2.putText(map_display, f"({map_x},{map_y})", (map_x+10, map_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 6. CREATE THE DASHBOARD (Split Screen)
    # Resize camera to fit nicely next to the 600x600 map
    frame_resized = cv2.resize(frame, (600, 600))
    dashboard = np.hstack((frame_resized, map_display))

    cv2.imshow("Safety Digital Twin (Press 'q' to quit)", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()