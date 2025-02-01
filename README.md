# 🚀 YOLOv8 Object Detection Project

Welcome to the **YOLOv8 Object Detection Project**! This repository encompasses the complete workflow of training, exporting, optimizing, and deploying a YOLOv8 model for efficient object detection. Whether you're a beginner or an experienced practitioner, this guide will help you navigate through each step seamlessly.

---

## 📝 Table of Contents

- [🚀 YOLOv8 Object Detection Project](#-yolov8-object-detection-project)
  - [📝 Table of Contents](#-table-of-contents)
  - [📖 Introduction](#-introduction)
  - [🔗 Project Links](#-project-links)
  - [📁 Repository Structure](#-repository-structure)
  - [📊 Data Preparation](#-data-preparation)
    - [📁 Dataset](#-dataset)
    - [📝 Data Configuration](#-data-configuration)
  - [🤖 Model](#-model)
    - [YOLOv8 Architecture](#yolov8-architecture)
    - [Model Files](#model-files)
  - [🏋️‍♂️ Training](#️️-training)
    - [🛠️ Setup](#️-setup)
- [Define project directories](#define-project-directories)
- [Initialize and train the model](#initialize-and-train-the-model)
    - [🎯 Objectives](#-objectives)
  - [⚙️ Inference \& Cross-Verification](#️-inference--cross-verification)
    - [Inference Process](#inference-process)
      - [Steps:](#steps)
    - [Example Explanation](#example-explanation)
  - [💡 Optimization Techniques](#-optimization-techniques)
  - [🔧 Troubleshooting](#-troubleshooting)
    - [Common Issues](#common-issues)
    - [Tips](#tips)
  - [👥 Contributing](#-contributing)
  - [📜 License](#-license)
  - [📫 Contact](#-contact)

---

## 📖 Introduction

This project leverages the **YOLOv8** architecture to perform object detection tasks. The workflow includes:

- **Data Preparation**: Handling various image formats, including HEIC.
- **Model Training**: Fine-tuning the YOLOv8 model on custom datasets.
- **Model Exporting**: Converting the trained model to ONNX format for enhanced interoperability.
- **Inference**: Running efficient object detection using ONNX Runtime.
- **Optimization**: Reducing model size and improving performance through simplification and quantization.

---

## 🔗 Project Links

- **Documentation**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/yolov8/)

---

## 📁 Repository Structure

```
YOLOv8_Object_Detection_Project/
├── data/ 
│ └── object_detection_yolov8/ 
│ └── data.yaml 
├── Real_photos/ 
│ └── IMG_3766.HEIC 
├── trained_models/ 
│ ├── yolov8_large_model.onnx 
│ ├── yolov8_large_model_simplified.onnx 
│ └── yolov8_large_model_quantized.onnx 
├── runs/ 
│ └── detect/ 
│ └── train5/ 
│ └── weights/ 
│ ├── best.pt 
│ └── best.onnx 
├── app/ 
│ ├── main.py 
│ ├── inference.py 
│ └── preprocess.py 
├── README.md 
└── requirements.txt
```

---

## 📊 Data Preparation

For this project, we utilized a combination of **synthetic data** and **real-world photographs** to train and evaluate our YOLOv8 object detection model. The synthetic data comprised high-resolution screenshots of objects captured from various angles, ensuring a diverse representation of each class. To streamline the annotation process and enhance data quality, we uploaded the dataset to [Roboflow Dataset](https://roboflow.com/), a powerful platform for managing and augmenting computer vision datasets.

Within Roboflow, we meticulously annotated each image with precise bounding boxes around the objects of interest, ensuring consistency and accuracy across the dataset. To further improve the model's ability to generalize to real-world scenarios, we applied a series of advanced augmentation techniques available in Roboflow's augmentation pipeline. These augmentations included:

* **Rotation:** Randomly rotating images to simulate different object orientations and viewpoints.
* **Scaling:** Varying the size of objects within images to mimic different distances and sizes encountered in real-world environments.
* **Brightness and Contrast Adjustment:** Altering the lighting conditions to account for varying illumination in different settings.
* **Flipping:** Horizontally and vertically flipping images to increase the diversity of object appearances.
* **Noise Addition:** Introducing random noise to emulate real-world image imperfections and enhance the model's robustness against noisy inputs.

### 📁 Dataset

- **Location**: `/content/drive/MyDrive/Project/Real photos/`
- **Format**: Includes various image formats such as `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, and `.heic`.
- **Description**: A diverse set of images used to train and evaluate the YOLOv8 model for object detection.

### 📝 Data Configuration

- **File**: `data.yaml`
- **Contents**:

  ```yaml
  train: ../train/images
  val: ../valid/images
  test: ../test/images

  nc: 5
  names: ['Bottle', 'HelloPanda', 'Potato_sticks', 'Rita_zero_sugar', 'Salt']

  roboflow:
    workspace: project-c6dcl
    project: object_detection-xngop
    version: 2
    license: CC BY 4.0
    url: https://universe.roboflow.com/project-c6dcl/object_detection-xngop/dataset/2
  ```

## 🤖 Model

### YOLOv8 Architecture

* **Version** : Ultralytics YOLOv8 8.3.70
* **Parameters** : 43,610,463
* **GFLOPs** : 164.8
* **Layers** : 268

### Model Files

* **PyTorch Model** : `best.pt` (86 MB)
* **ONNX Models** : `yolov8_large_model.onnx` (166.6 MB)

## 🏋️‍♂️ Training

### 🛠️ Setup

1. **Install Dependencies** :

   ```bash
   pip install ultralytics
   ```
2. **Training Command** :

   ```bash
   from ultralytics import YOLO
   project_dir = '/content/drive/MyDrive/Project'
   data_config = os.path.join(project_dir, 'data', 'object_detection_yolov8', 'data.yaml')

   model = YOLO('yolov8l.pt')  # Use YOLOv8 large pre-trained weights
   model.train(data=data_config, epochs=100, patience=10)
   ```

# Define project directories

```bash
   project_dir = '/content/drive/MyDrive/Project'
   data_config = os.path.join(project_dir, 'data', 'object_detection_yolov8', 'data.yaml')
```

# Initialize and train the model

```bash
   model = YOLO('yolov8l.pt')  # Use YOLOv8 large pre-trained weights
   model.train(data=data_config, epochs=100, patience=10)
```

### 🎯 Objectives

* **Prevent Overfitting** : Implement early stopping based on validation performance.
* **Regularization** : Apply weight decay and dropout where applicable.
* **Cross-Validation** : Utilize K-Fold cross-validation to ensure model generalization.

## ⚙️ Inference & Cross-Verification

### Inference Process

We perform inference on sample JPG images from the `Real photos` folder and visualize the results to verify that both our custom-trained and Roboflow-trained models achieve similar accuracy.

#### Steps:

1. **Model Loading** : Load the trained YOLOv8 model(s).
2. **Image Selection** : Choose sample images (JPG) from the dataset.
3. **Run Inference** : Obtain predictions (bounding boxes, class labels, confidence scores).
4. **Visualization** : Display the annotated images with bounding boxes and labels.
5. **Comparison** : Cross-check the predictions from both models side-by-side.

### Example Explanation

* **Custom Model** : Loaded from `/runs/detect/train5/weights/best.pt`.
* **Roboflow Model** : Loaded from its respective path.
* **Visualization** : Both models’ outputs are displayed on the same test image for comparison.
* **Prediction Details** : The console prints out the detected class names, confidence scores, and bounding box coordinates for each model.

---

## 💡 Optimization Techniques

To further improve detection accuracy and performance, consider the following:

* **Data Augmentation** : We trained both with and without Augmentation Rotate, scale, adjust brightness/contrast, and flip images.
* **Hyperparameter Tuning** : Adjust learning rates, batch sizes, and image sizes.
* **Transfer Learning** : Leverage pre-trained models and fine-tune them on your dataset.
* **Advanced Post-Processing** : Apply Non-Maximum Suppression (NMS) with fine-tuned IoU thresholds.
* **ONNX Optimization** : Simplify and quantize the ONNX model to reduce size and improve inference speed.

---

## 🔧 Troubleshooting

### Common Issues

* **File Not Found Errors** : Ensure that the paths to the model and data are correct.
* **Conversion Errors** : Verify that HEIC files are valid and that required packages (`pyheif`, `Pillow`) are installed.
* **Inference Crashes** : Check resource availability and consider using GPU acceleration in Colab.

### Tips

* **Runtime Environment** : Use Google Colab’s GPU accelerator for faster inference.
* **Logging** : Monitor outputs and error messages for debugging.
* **Community Support** : Consult the [Ultralytics Forums](https://community.ultralytics.com/) or [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) for help.

---

## 👥 Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository**
2. **Create a New Branch** : `git checkout -b feature/YourFeature`
3. **Commit Your Changes** : `git commit -m "Add Your Feature"`
4. **Push and Open a Pull Request**

---

## 📜 License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 📫 Contact

* **Your Name** : [nawazktk99@gmail.com](Email Me "Email Me")
* **GitHub** : [https://www.github.com/Aliktk/object_detection_yolov8](Repo)

---

✨ *Thank you for checking out the YOLOv8 Object Detection Project! Happy detecting!*

---
