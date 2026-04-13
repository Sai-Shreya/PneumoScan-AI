# Pneumonia Detection from Chest X-Rays

This project implements an end-to-end deep learning system for detecting pneumonia from chest X-ray images. It includes data preprocessing, custom CNN models, transfer learning, and a web-based deployment interface.

## Project Structure

- `src/`: Modular Python scripts for different components.
  - `config.py`: Central configuration and paths.
  - `data_loader.py`: Image loading and augmentation logic.
  - `model_builder.py`: CNN architecture definitions.
  - `trainer.py`: Training loop and class weight handling.
  - `evaluator.py`: Performance metrics and visualization.
  - `gradcam.py`: Explainability (heatmap generation).
- `app.py`: Streamlit web application.
- `main.py`: Entry point for the training and evaluation pipeline.
- `exports/`: Stores saved models (`.h5`) and evaluation graphs.
- `requirements.txt`: Project dependencies.

## How to Run

### 1. Training the Models
To train both the custom CNN and the ResNet50 models, run:
```bash
python main.py
```
*Note: Since training is done on CPU, it may take several hours. Results will be saved in the `exports/` folder.*

### 2. Running the Web App
Once the models are trained, you can launch the interactive dashboard:
```bash
streamlit run app.py
```

## Features

- **Transfer Learning**: Uses ResNet50 pre-trained on ImageNet for high-accuracy feature extraction.
- **Data Augmentation**: Robust preprocessing to handle variations in X-ray quality and orientation.
- **Explainability**: Integrated Grad-CAM to highlight exactly where the model is looking to make a diagnosis.
- **Metrics**: Detailed evaluation including Confusion Matrix, F1-Score, and ROC-AUC.

## Disclaimer
This project is for demonstration purposes only. It is not intended for use in clinical environments or for providing medical diagnoses.
