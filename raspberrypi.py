import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from gpiozero import AngularServo
import time
import os

# ── Display detection ──────────────────────────────────
DISPLAY_AVAILABLE = os.environ.get("DISPLAY") is not None
if DISPLAY_AVAILABLE:
    print("Display detected — live window enabled.")
else:
    print("No display — running in silent sorting mode.")

# ── Model ──────────────────────────────────────────────
MODEL_PATH = "/home/momentum/waste_sorter/models/plastic_classifier_compatible.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Servo ──────────────────────────────────────────────
servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)

SERVO_IDLE              =   0
SERVO_NON_BIODEGRADABLE =  90
SERVO_BIODEGRADABLE     = -90

def go_idle():
    servo.angle = SERVO_IDLE
    time.sleep(0.5)
    servo.detach()
    print("  [Servo] Idle — lid closed")

def sort_item(label):
    if label == "NON-BIODEGRADABLE":
        print("  [Servo] Opening to NON-BIODEGRADABLE bin...")
        servo.angle = SERVO_NON_BIODEGRADABLE
    else:
        print("  [Servo] Opening to BIODEGRADABLE bin...")
        servo.angle = SERVO_BIODEGRADABLE
    time.sleep(1.5)
    go_idle()

# ── Motion Detection ───────────────────────────────────
MOTION_THRESHOLD = 25
MOTION_MIN_AREA  = 3000

def object_present(prev_frame, curr_frame):
    if prev_frame is None:
        return False
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    diff      = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    changed   = cv2.countNonZero(thresh)
    return changed > MOTION_MIN_AREA

# ── Classification ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55

def classify_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(
        output_details[0]['index'])[0][0]
    return prediction

def show_frame(frame, label="", confidence=0, color=(255,255,255), status=""):
    if not DISPLAY_AVAILABLE:
        return
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(display, f"{label} ({confidence*100:.1f}%)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if status:
        cv2.putText(display, status,
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
    h, w = display.shape[:2]
    cx, cy = w // 2, h // 2
    box = 150
    cv2.rectangle(display, (cx-box, cy-box),
                  (cx+box, cy+box), color, 2)
    cv2.imshow("AI Waste Sorter", display)
    cv2.waitKey(1)

# ── Camera ─────────────────────────────────────────────
print("Initialising camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

go_idle()
print("System ready — waiting for object...\n")
if DISPLAY_AVAILABLE:
    print("Press Q in the window to quit.\n")
else:
    print("Press Ctrl+C to stop.\n")

prev_frame = None

try:
    while True:
        curr_frame = picam2.capture_array()

        # Stage 1 — Motion Detection
        if object_present(prev_frame, curr_frame):
            print("Object detected — classifying...")
            time.sleep(0.4)

            stable_frame = picam2.capture_array()
            prediction   = classify_frame(stable_frame)

            if prediction >= CONFIDENCE_THRESHOLD:
                label      = "NON-BIODEGRADABLE"
                confidence = prediction
                color      = (0, 0, 255)      # red
            elif prediction <= (1 - CONFIDENCE_THRESHOLD):
                label      = "BIODEGRADABLE"
                confidence = 1 - prediction
                color      = (0, 255, 0)      # green
            else:
                label      = "UNCERTAIN"
                confidence = max(prediction, 1 - prediction)
                color      = (0, 255, 255)    # yellow

            print(f"Result: {label} ({confidence*100:.1f}% confidence)")
            show_frame(stable_frame, label, confidence, color, "SORTING...")

            if label in ["NON-BIODEGRADABLE", "BIODEGRADABLE"]:
                sort_item(label)
            else:
                print("  Uncertain — no sorting action.")
                go_idle()

            print("Waiting for next object...\n")
            time.sleep(2)
            prev_frame = None
            continue

        else:
            show_frame(curr_frame, status="WAITING FOR OBJECT...")

        prev_frame = curr_frame

        if DISPLAY_AVAILABLE:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    picam2.stop()
    if DISPLAY_AVAILABLE:
        cv2.destroyAllWindows()
    go_idle()
    print("System stopped cleanly.")