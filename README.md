# Plastic Waste Classifier

## Overview

This folder contains the trained machine learning model for the AI-Based Plastic Waste Recognition and Automated Sorting System**. The model was built using transfer learning on MobileNetV2 and is designed to classify waste items as either Plastic or Non-Plastic in real time on a Raspberry Pi 3.

## Files

| File | Size | Description |

| `plastic_classifier.h5` | ~13 MB | Full Keras model — used for testing on PC/laptop 
| `plastic_classifier.tflite` | 2.41 MB | Compressed TFLite model — used on Raspberry Pi 
| `train_model.py` | — | Script used to train the model |
| `convert_to_tflite.py` | — | Script used to convert `.h5` to `.tflite` |
| `test_webcam.py` | — | Script to test the model live on a PC webcam |
| `check_data.py` | — | Script to clean the dataset before training |

## Model Details

| Property | Value |

| Base Architecture | MobileNetV2 (pretrained on ImageNet) 
| Training Method | Transfer Learning (2-stage) 
| Input Size | 224 x 224 pixels (RGB) 
| Output | Single sigmoid neuron (probability of Plastic) 
| Classes | `non_plastic` (index 0), `plastic` (index 1) 
| Final Test Accuracy | 92.4% 
| TFLite Model Size | 2.41 MB 
| Quantization | Dynamic range quantization 


## Dataset

| Class | Images Used 

| plastic | 6,143 images 
| non_plastic | 5,718 image
| Total ----> 11,861 images

Sources used:
- Original binary plastic/non-plastic dataset
- Garbage Classification v2 (Kaggle — sumn2u)

Preprocessing applied:
- XML annotation files removed
- Corrupt/non-image files removed
- Images resized to 224x224 at training time
- MobileNetV2 `preprocess_input` normalization (scales pixels to [-1, 1])
- Data augmentation: random horizontal flip, rotation (±20°), zoom (±10%)



## Training Process

Training was done in two stages on a Windows PC (CPU only) using TensorFlow 2.21.0.

Stage 1 — Feature Extraction (10 epochs)
- MobileNetV2 base frozen
- Only the new classification head trained
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Crossentropy

Stage 2 — Fine-Tuning (5 epochs)
- Top layers of MobileNetV2 unfrozen (from layer 100 onward)
- Very low learning rate to avoid destroying pretrained features
- Learning rate: 0.00001
- Optimizer: Adam


## How to Use

### On your PC (webcam test)

Make sure your virtual environment is active:

```bash
.venv\Scripts\activate
```

Then run:

```bash
python test_webcam.py
```

Hold an object in front of the camera inside the rectangle shown on screen:
- 🔴 RED label = PLASTIC — goes in the plastic bin
- 🟢 GREEN label = NON-PLASTIC — goes in the non-plastic bin
- 🟡 YELLOW label = UNCERTAIN — model isn't confident enough to decide

Press Q to quit the webcam window.


## Project Folder Structure

```
waste_sorter/
├── .venv/                        # Python virtual environment (Windows)
├── dataset/
│   ├── plastic/                  # 6,143 training images
│   └── non_plastic/              # 5,718 training images
├── models/
│   ├── plastic_classifier.h5     # Full Keras model
│   └── plastic_classifier.tflite # Compressed TFLite model for Pi
├── check_data.py                 # Dataset cleaning script
├── train_model.py                # Model training script
├── convert_to_tflite.py          # TFLite conversion script
├── test_webcam.py                # PC webcam testing script
└── README.md                     
```

## Requirements

### PC (training and testing)
```
tensorflow==2.21.0
numpy
pillow
opencv-python
matplotlib
```

## Known Limitations

- Model was trained on studio/clean-background images — performance may vary under unusual lighting or cluttered backgrounds
- Transparent plastic items (clear bags, cling film) may occasionally be misclassified
- Model handles one item at a time — multiple overlapping objects in frame will reduce accuracy
- Confidence threshold may need tuning per deployment environment

---

## Author Notes

Model trained June 2026. If retraining is needed (e.g. to add more classes or improve accuracy on specific items), run `train_model.py` with updated data in the `dataset/` folder and then re-run `convert_to_tflite.py` to generate a fresh `.tflite` file for the Pi.