# 🍽️ Food Image Authenticity Detector

A deep learning pipeline for detecting AI-generated, real, and AI-edited food images using EfficientNet-B3. This project provides a complete end-to-end solution from training to deployment with a Streamlit web interface.

## 📋 Overview

This project classifies food images into three categories:
- **Real**: Authentic, unedited food photographs
- **AI**: AI-generated food images
- **Edited**: Real images that have been AI-edited or enhanced

The model is built on EfficientNet-B3, a state-of-the-art convolutional neural network architecture, fine-tuned for food image authenticity detection.

## ✨ Features

- 🔥 **EfficientNet-B3** based classifier for high accuracy
- 📦 **Custom H5 dataset loader** with PyTorch DataLoader support
- 🧠 **Complete training pipeline** with configurable hyperparameters
- 📈 **Batch-wise accuracy visualization** for model evaluation
- 🧪 **Accuracy evaluation script** to measure model performance
- 🎯 **Real-time inference** on new food images
- 🌐 **Streamlit web app** for easy image classification
- 🧹 **Clean and modular code structure**

## 🛠️ Tech Stack

- **Python 3**
- **PyTorch** - Deep learning framework
- **TorchVision** - Pre-trained models and transforms
- **Albumentations** - Advanced image augmentations
- **OpenCV** - Image processing
- **Streamlit** - Web interface
- **Matplotlib** - Visualization
- **NumPy** - Numerical operations
- **h5py** - HDF5 dataset handling

## 📂 Project Structure

```
Food-Detection/
│
├── dataset.py          # H5 dataset loader and DataLoader creation
├── model.py            # EfficientNet-B3 classifier model definition
├── train.py            # Training script with configurable parameters
├── inference.py        # Single image prediction function
├── checkACC.py         # Overall model accuracy evaluation
├── plot.py             # Batch-wise accuracy visualization
├── app.py              # Streamlit web application
│
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aryankhunt30/Food-Detection.git
   cd Food-Detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses an H5 (HDF5) dataset file with the following structure:
- `images`: Image data array
- `category`: Label indices for each image
- `category_names`: Class name mappings

**Note**: The dataset file (`food_c101_n1000_r384x384x3.h5`) is not included in this repository due to size limitations. You'll need to provide your own dataset file or download it separately.

## 🏋️ Training the Model

To train the model, run:

```bash
python train.py
```

### Training Parameters

You can modify the training parameters in `train.py`:

```python
train(
    h5_path="food_c101_n1000_r384x384x3.h5",
    batch_size=32,      # Batch size for training
    epochs=5,           # Number of training epochs
    lr=1e-4             # Learning rate
)
```

The trained model will be saved as `model_food.pth`.

## 📈 Model Evaluation

### Check Overall Accuracy

To evaluate the model's accuracy on the entire dataset:

```bash
python checkACC.py
```

**Note**: Update the dataset path in `checkACC.py` to match your local file location.

### Visualize Batch-wise Accuracy

To generate an accuracy curve showing performance across batches:

```bash
python plot.py
```

This will display a graph showing cumulative accuracy over batches.

## 🎯 Inference

### Single Image Prediction

Use the `inference.py` module to predict on a single image:

```python
from inference import predict

results = predict("path/to/image.jpg")
print(results)
# Output: {'real': 0.85, 'ai': 0.10, 'edited': 0.05}
```

### Web Application

Launch the Streamlit web app for interactive image classification:

```bash
streamlit run app.py
```

Then open your browser to the URL shown (typically `http://localhost:8501`) and upload food images to get real-time predictions.

## 📝 Usage Examples

### Training Example
```python
from train import train

train(
    h5_path="food_c101_n1000_r384x384x3.h5",
    batch_size=32,
    epochs=10,
    lr=1e-4
)
```

### Inference Example
```python
from inference import predict

# Predict on an image
results = predict("food_image.jpg", model_path="model_food.pth")
print(f"Real: {results['real']:.2%}")
print(f"AI: {results['ai']:.2%}")
print(f"Edited: {results['edited']:.2%}")
```

## 🔧 Configuration

### Model Architecture

The model uses EfficientNet-B3 with:
- **Base**: Pre-trained EfficientNet-B3 weights
- **Classifier**: Custom linear layer for 3-class classification
- **Input Size**: 300x300 pixels (configurable in inference)

### Data Augmentation

You can add data augmentation in `train.py`:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
])
```

## 📦 Model Files

**Note**: Large model and dataset files are excluded from this repository:
- `model_food.pth` - Trained model weights (not tracked in git)
- `food_c101_n1000_r384x384x3.h5` - Dataset file (not tracked in git)

These files remain on your local machine. To share them:
- Use **Git LFS** for version control
- Upload to **cloud storage** (Google Drive, Dropbox) with download links
- Use **GitHub Releases** for distribution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Aryan Khunt**
- Machine Learning Engineer
- Los Angeles, CA

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch team for the deep learning framework
- Streamlit for the web framework

---

⭐ If you find this project helpful, please consider giving it a star!
