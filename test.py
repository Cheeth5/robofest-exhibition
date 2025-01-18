from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('new.pt')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change the index for other cameras

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Visualize the results (if your YOLO version supports rendering)
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
