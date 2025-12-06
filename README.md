# AI-Food-detection-

🍽️ Food Image Classification using EfficientNet-B3

A Deep Learning Pipeline for Real-Time Food Detection

⸻

📌 Overview

This project builds a complete machine learning pipeline to classify food images using EfficientNet-B3.
It covers everything from loading a custom .h5 dataset, training the model, evaluating performance, generating accuracy curves, and performing real-time inference.

The goal was to create a simple but powerful end-to-end food image classification system that can be extended for production use.

⸻

🚀 Features
	•	🔥 EfficientNet-B3–based food classifier
	•	📦 Custom .h5 dataset loader with PyTorch
	•	🧠 Full training pipeline (loss, optimizer, epochs)
	•	📈 Batch-wise accuracy visualization graph
	•	🧪 Easy evaluation script to measure model accuracy
	•	🎯 Real-time inference on new food images
	•	🌐 Optional demo app (Streamlit/Flask)
	•	🧹 Clean and modular code structure

⸻

🧠 Tech Stack
	•	Python 3
	•	PyTorch
	•	TorchVision
	•	Albumentations
	•	Matplotlib
	•	NumPy
	•	h5py

📂 Project Structure

Food-Detection/
│
├── dataset.py            # Loads images & labels from .h5 dataset
├── model.py              # EfficientNet-B3 classifier model
├── train.py              # Training loop
├── inference.py          # Predicts class for new images
├── checkACC.py           # Computes overall model accuracy
├── plot.py               # Accuracy graph (batch-wise)
├── app.py                # Simple UI for predictions (optional)
│
├── model_food.pth        # Saved trained model weights
├── requirements.txt      # Dependencies
└── README.md             # Project documentation


🏋️ Training the Model

To train the model: python train.py

📊 Accuracy Curve (Visualization)

To generate a batch-wise accuracy graph:python plot.py

Example interpretation:
	•	Batch 0 → 97.3% accuracy
	•	Batch 1 → 97.8%
	•	Batch 2 → 97.4%
	•	Batch 3 → 98.1%
	•	Overall accuracy ~97.5%

**🙌 Author

Aryan Khunt
Machine Learning Engineer
Los Angeles, CA**










