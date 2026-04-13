import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import os
from src.config import EXPORT_DIR

def plot_history(history, title, save_path):
    """
    Plots training and validation accuracy/loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def evaluate_on_test(model, test_gen, title, save_prefix):
    """
    Evaluates the model on the test set and saves plots.
    """
    # Predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{save_prefix}_cm.png")
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'])
    print(f"Classification Report for {title}:\n", report)
    with open(f"{save_prefix}_report.txt", "w") as f:
        f.write(report)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_prefix}_roc.png")
    plt.close()
    
    return roc_auc
