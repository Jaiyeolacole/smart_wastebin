import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("plastic_classifier.h5")
print("Model loaded successfully.")

# Convert to TensorFlow Lite with dynamic range quantization
# This shrinks the model from ~13 MB down to ~3-4 MB
# and makes it run faster on the Raspberry Pi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the .tflite file
with open("plastic_classifier.tflite", "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"TFLite model saved as plastic_classifier.tflite")
print(f"Model size: {size_mb:.2f} MB")
print("Ready to copy to Raspberry Pi.")