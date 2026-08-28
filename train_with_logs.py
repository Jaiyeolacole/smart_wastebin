import tensorflow as tf
import csv
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.30, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.30, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)

val_batches = tf.data.experimental.cardinality(val_test_ds)
test_ds = val_test_ds.take(val_batches // 2)
val_ds = val_test_ds.skip(val_batches // 2)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.06),
    tf.keras.layers.RandomZoom(0.1),
])

def prepare(ds, augment=False, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    ds = ds.map(lambda x, y: (preprocess_input(x), y))
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = prepare(train_ds, augment=True, shuffle=True)
val_ds   = prepare(val_ds)
test_ds  = prepare(test_ds)

# Build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model   = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

print("\n=== Stage 1: Feature Extraction ===")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Fine-tune
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

print("\n=== Stage 2: Fine-Tuning ===")
history2 = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Final evaluation
print("\n=== Final Evaluation on Test Set ===")
results = model.evaluate(test_ds, verbose=1)
print(f"Test Loss:      {results[0]:.4f}")
print(f"Test Accuracy:  {results[1]:.4f}")
print(f"Test Precision: {results[2]:.4f}")
print(f"Test Recall:    {results[3]:.4f}")
f1 = 2 * (results[2] * results[3]) / (results[2] + results[3])
print(f"Test F1 Score:  {f1:.4f}")

# Save all logs to CSV
with open("training_logs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Stage", "Epoch", "Loss", "Accuracy",
                     "Precision", "Recall",
                     "Val_Loss", "Val_Accuracy",
                     "Val_Precision", "Val_Recall"])
    for i in range(len(history1.history["loss"])):
        writer.writerow([
            "Stage1", i+1,
            f"{history1.history['loss'][i]:.4f}",
            f"{history1.history['accuracy'][i]:.4f}",
            f"{history1.history['precision'][i]:.4f}",
            f"{history1.history['recall'][i]:.4f}",
            f"{history1.history['val_loss'][i]:.4f}",
            f"{history1.history['val_accuracy'][i]:.4f}",
            f"{history1.history['val_precision'][i]:.4f}",
            f"{history1.history['val_recall'][i]:.4f}",
        ])
    for i in range(len(history2.history["loss"])):
        writer.writerow([
            "Stage2", i+1,
            f"{history2.history['loss'][i]:.4f}",
            f"{history2.history['accuracy'][i]:.4f}",
            f"{history2.history['precision'][i]:.4f}",
            f"{history2.history['recall'][i]:.4f}",
            f"{history2.history['val_loss'][i]:.4f}",
            f"{history2.history['val_accuracy'][i]:.4f}",
            f"{history2.history['val_precision'][i]:.4f}",
            f"{history2.history['val_recall'][i]:.4f}",
        ])

# Save test results
with open("test_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Test Loss", f"{results[0]:.4f}"])
    writer.writerow(["Test Accuracy", f"{results[1]:.4f}"])
    writer.writerow(["Test Precision", f"{results[2]:.4f}"])
    writer.writerow(["Test Recall", f"{results[3]:.4f}"])
    writer.writerow(["Test F1 Score", f"{f1:.4f}"])

model.save("plastic_classifier.h5")
print("\nLogs saved to training_logs.csv and test_results.csv")
print("Model saved as plastic_classifier.h5")