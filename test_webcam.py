import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model("plastic_classifier.h5")
print("Model loaded. Starting webcam...")

# Class names must match your dataset folder order (alphabetical)
# dataset has 'non_plastic' and 'plastic' — alphabetically non_plastic=0, plastic=1
CLASS_NAMES = ["non_plastic", "plastic"]
CONFIDENCE_THRESHOLD = 0.7  # only show result if model is at least 70% confident

# Open webcam (0 = default built-in webcam, try 1 if this doesn't work)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing 0 to 1 in VideoCapture(0).")
    exit()

print("Webcam open. Hold an object in front of the camera.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame for the model
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))

    # Run inference
    prediction = model.predict(img, verbose=0)[0][0]

    # prediction is probability of being 'plastic' (class index 1)
    if prediction >= CONFIDENCE_THRESHOLD:
        label = "PLASTIC"
        confidence = prediction
        color = (0, 0, 255)      # red box for plastic
    elif prediction <= (1 - CONFIDENCE_THRESHOLD):
        label = "NON-PLASTIC"
        confidence = 1 - prediction
        color = (0, 255, 0)      # green box for non-plastic
    else:
        label = "UNCERTAIN"
        confidence = max(prediction, 1 - prediction)
        color = (0, 255, 255)    # yellow for uncertain

    # Draw result on the frame
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Draw a rectangle in the centre showing what the model is "looking at"
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    box = 200
    cv2.rectangle(frame, (cx - box, cy - box), (cx + box, cy + box), color, 2)
    cv2.putText(frame, "Hold object here", (cx - box, cy - box - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Plastic Classifier - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")