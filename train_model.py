import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/"

# Load and split data: 70% train, 30% held out (split again below into val/test)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.30, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.30, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)  #print ['non_plastic', 'plastic']

val_batches = tf.data.experimental.cardinality(val_test_ds)
test_ds = val_test_ds.take(val_batches // 2)
val_ds = val_test_ds.skip(val_batches // 2)

# Data augmentation 
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
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# Build model: MobileNetV2 base (pretrained on ImageNet) + small classifier head
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
)
base_model.trainable = False  # freeze for stage 1

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Stage 1: training classifier head (base frozen) ===")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Stage 2: unfreeze top layers and fine-tune at a low learning rate
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Stage 2: fine-tuning ===")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Final evaluation on held-out test set
loss, acc = model.evaluate(test_ds)
print(f"\nFinal test accuracy: {acc:.3f}")

model.save("plastic_classifier.h5")
print("Model saved as plastic_classifier.h5")