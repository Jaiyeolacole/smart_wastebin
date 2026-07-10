import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("plastic_classifier.h5")
print("Model loaded successfully.")

# Convert targeting older op versions compatible with tflite-runtime 2.14.0
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Force compatibility with older runtime versions
converter._experimental_lower_tensor_list_ops = False
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the .tflite file
with open("plastic_classifier_compatible.tflite", "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"Compatible TFLite model saved as plastic_classifier_compatible.tflite")
print(f"Model size: {size_mb:.2f} MB")