import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from gpiozero import AngularServo
import time

# ── Model ──────────────────────────────────────────────
MODEL_PATH = "/home/momentum/waste_sorter/models/plastic_classifier_compatible.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Servo ──────────────────────────────────────────────
servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)
SERVO_HOME        =   0   # neutral / centre
SERVO_PLASTIC     =  45   # plastic bin
SERVO_NON_PLASTIC = -45   # non-plastic bin

def sort_item(label):
    if label == "PLASTIC":
        print("  >> Sorting to PLASTIC bin")
        servo.angle = SERVO_PLASTIC
    else:
        print("  >> Sorting to NON-PLASTIC bin")
        servo.angle = SERVO_NON_PLASTIC
    time.sleep(1.5)        # hold position so item falls into bin
    servo.angle = SERVO_HOME
    time.sleep(0.5)        # return to home

# ── Camera ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55

print("Loading model and camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# Home the servo on startup
servo.angle = SERVO_HOME
print("System ready. Press Q to quit.\n")

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Preprocess for model
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Interpret result
    if prediction >= CONFIDENCE_THRESHOLD:
        label      = "PLASTIC"
        confidence = prediction
        color      = (0, 0, 255)    # red
    elif prediction <= (1 - CONFIDENCE_THRESHOLD):
        label      = "NON-PLASTIC"
        confidence = 1 - prediction
        color      = (0, 255, 0)    # green
    else:
        label      = "UNCERTAIN"
        confidence = max(prediction, 1 - prediction)
        color      = (0, 255, 255)  # yellow

    # Display
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(display, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    h, w = display.shape[:2]
    cx, cy = w // 2, h // 2
    box = 150
    cv2.rectangle(display, (cx-box, cy-box), (cx+box, cy+box), color, 2)
    cv2.putText(display, "Hold object here", (cx-box, cy-box-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("AI Waste Sorter", display)
    print(f"Result: {label} ({confidence*100:.1f}% confidence)")

    # Trigger servo if confident
    if label in ["PLASTIC", "NON-PLASTIC"]:
        sort_item(label)
        time.sleep(3)   # brief pause before next classification

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
servo.angle = SERVO_HOME
print("System stopped.")