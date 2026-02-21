import cv2
import numpy as np

# --- PART 1: GENERATE THE DIGITAL MAP ---
# We create a 600x600 white image with a Red Danger Zone
map_size = 600
map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255 # White background

# Draw Red Danger Zone (Coordinates 200,200 to 400,400)
cv2.rectangle(map_img, (200, 200), (400, 400), (0, 0, 255), -1) 
# Draw Green Border (Safe Zone boundary)
cv2.rectangle(map_img, (0, 0), (600, 600), (0, 255, 0), 5)

# Save the map image
cv2.imwrite("map_image.jpg", map_img)
print("✅ Created 'map_image.jpg'")

# --- PART 2: CALIBRATE CAMERA ---
points_camera = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_camera) < 4:
            points_camera.append([x, y])
            print(f"Captured Point {len(points_camera)}: {x}, {y}")

# Open Camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# If camera doesn't open, try index 1
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

print("\n--- INSTRUCTIONS ---")
print("1. A window will open showing your camera.")
print("2. Click the 4 markers on the floor in this EXACT order:")
print("   TOP-LEFT  ->  TOP-RIGHT  ->  BOTTOM-RIGHT  ->  BOTTOM-LEFT")
print("3. Press 'q' when done.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Draw circles where you clicked
    for i, pt in enumerate(points_camera):
        cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Calibration", frame)

    if len(points_camera) == 4:
        # Wait a moment to show the 4th dot, then close
        cv2.waitKey(500) 
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- PART 3: CALCULATE & SAVE MATRIX ---
if len(points_camera) == 4:
    pts_src = np.array(points_camera, dtype=float)
    
    # Map these to the 4 corners of our 600x600 digital map
    pts_dst = np.array([
        [0, 0],               # Top-Left
        [map_size, 0],        # Top-Right
        [map_size, map_size], # Bottom-Right
        [0, map_size]         # Bottom-Left
    ], dtype=float)

    h_matrix, status = cv2.findHomography(pts_src, pts_dst)
    np.save("calibration_matrix.npy", h_matrix)
    print("\n✅ SUCCESS: 'calibration_matrix.npy' saved!")
    print("You are ready for the final step.")
else:
    print("\n❌ FAILED: You didn't click 4 points.")