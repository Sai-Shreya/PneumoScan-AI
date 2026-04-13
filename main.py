import os
import tensorflow as tf
from src.config import CUSTOM_MODEL_PATH, PRETRAINED_MODEL_PATH, EXPORT_DIR
from src.data_loader import get_data_generators
from src.model_builder import build_custom_cnn, build_transfer_learning_model
from src.trainer import train_model
from src.evaluator import plot_history, evaluate_on_test

def run_pipeline():
    print("Starting Pneumonia Detection Pipeline...")
    
    # 1. Load Data
    train_gen, val_gen, test_gen = get_data_generators()
    
    # 2. Train Custom CNN
    print("\n--- Training Custom CNN ---")
    custom_model = build_custom_cnn()
    custom_history = train_model(custom_model, train_gen, val_gen, CUSTOM_MODEL_PATH)
    
    # Evaluate Custom CNN
    plot_history(custom_history, "Custom CNN", os.path.join(EXPORT_DIR, "custom_cnn_history.png"))
    evaluate_on_test(custom_model, test_gen, "Custom CNN", os.path.join(EXPORT_DIR, "custom_cnn"))
    
    # 3. Train Transfer Learning (ResNet50)
    print("\n--- Training Transfer Learning (ResNet50) ---")
    resnet_model = build_transfer_learning_model()
    resnet_history = train_model(resnet_model, train_gen, val_gen, PRETRAINED_MODEL_PATH)
    
    # Evaluate Transfer Learning
    plot_history(resnet_history, "ResNet50", os.path.join(EXPORT_DIR, "resnet50_history.png"))
    evaluate_on_test(resnet_model, test_gen, "ResNet50", os.path.join(EXPORT_DIR, "resnet50"))
    
    print("\nPipeline execution completed successfully. Models and plots saved in 'exports/' directory.")

if __name__ == "__main__":
    run_pipeline()
