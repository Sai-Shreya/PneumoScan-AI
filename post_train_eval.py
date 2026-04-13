"""
post_train_eval.py
──────────────────
Runs after training is complete.
Produces:
  - Confusion matrices for Custom CNN and ResNet50
  - ROC curves
  - Side-by-side comparison bar chart
  - 6 sample Grad-CAM predictions (3 PNEUMONIA, 3 NORMAL)
  - A summary text report
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from PIL import Image

from src.config import (CUSTOM_MODEL_PATH, PRETRAINED_MODEL_PATH,
                        EXPORT_DIR, IMG_SIZE)
from src.data_loader import get_data_generators
from src.gradcam import get_gradcam_heatmap, save_and_display_gradcam, find_last_conv_layer

sns.set_theme(style="darkgrid")
REPORTS_DIR = os.path.join(EXPORT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def evaluate_model(model, test_gen, name):
    """Return y_true, y_pred, y_probs for a model on the test set."""
    test_gen.reset()
    y_probs = model.predict(test_gen, verbose=1).flatten()
    y_pred  = (y_probs > 0.5).astype(int)
    y_true  = test_gen.classes
    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Pneumonia'],
                yticklabels=['Normal','Pneumonia'], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual');  ax.set_xlabel('Predicted')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc(y_true, y_probs, title, save_path, color='darkorange'):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return roc_auc


def comparison_bar(metrics_cnn, metrics_res, save_path):
    """Bar chart comparing CNN vs ResNet50 on key metrics."""
    labels  = ['Accuracy','Precision','Recall','F1-Score','AUC']
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    b1 = ax.bar(x - w/2, metrics_cnn, w, label='Custom CNN',
                color='#4facfe', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, metrics_res, w, label='ResNet50',
                color='#f093fb', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0, 1.12]); ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Custom CNN vs ResNet50',
                 fontsize=14, fontweight='bold')
    ax.legend()
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def gradcam_sample_grid(model, name):
    """Generate a 2×3 grid of Grad-CAM predictions on test images."""
    test_p = r"c:\Users\mdkan\OneDrive\Desktop\nndl\chest_xray\test\PNEUMONIA"
    test_n = r"c:\Users\mdkan\OneDrive\Desktop\nndl\chest_xray\test\NORMAL"
    p_imgs = [os.path.join(test_p, f) for f in os.listdir(test_p) if f.lower().endswith(('.jpeg','.jpg','.png'))]
    n_imgs = [os.path.join(test_n, f) for f in os.listdir(test_n) if f.lower().endswith(('.jpeg','.jpg','.png'))]
    samples = random.sample(p_imgs, 3) + random.sample(n_imgs, 3)
    true_labels = ['PNEUMONIA']*3 + ['NORMAL']*3

    last_conv = find_last_conv_layer(model)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'{name} — Grad-CAM Sample Predictions', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.3)

    for idx, (img_path, true_label) in enumerate(zip(samples, true_labels)):
        img_pil = Image.open(img_path).convert('RGB')
        img_arr = np.array(img_pil.resize(IMG_SIZE)) / 255.0
        img_inp = np.expand_dims(img_arr, 0).astype(np.float32)

        pred_score = float(model.predict(img_inp, verbose=0)[0][0])
        pred_label = "PNEUMONIA" if pred_score > 0.5 else "NORMAL"
        confidence = pred_score if pred_score > 0.5 else 1 - pred_score

        # Grad-CAM
        heatmap = get_gradcam_heatmap(model, img_inp, last_conv)
        tmp   = f"_tmp_sample_{idx}.jpg"
        tmpc  = f"_tmp_cam_{idx}.jpg"
        img_pil.save(tmp)
        save_and_display_gradcam(tmp, heatmap, tmpc)
        cam_img = Image.open(tmpc)

        # Original row
        ax_orig = fig.add_subplot(gs[0, idx])
        ax_orig.imshow(img_pil)
        ax_orig.set_title(f"True: {true_label}", fontsize=9)
        ax_orig.axis('off')

        # CAM row
        ax_cam = fig.add_subplot(gs[1, idx])
        ax_cam.imshow(cam_img)
        color = 'green' if pred_label == true_label else 'red'
        ax_cam.set_title(f"Pred: {pred_label}\n({confidence*100:.1f}%)",
                         fontsize=9, color=color)
        ax_cam.axis('off')

        # Cleanup
        for t in [tmp, tmpc]:
            if os.path.exists(t): os.remove(t)

    save_path = os.path.join(REPORTS_DIR, f"{name.replace(' ','_')}_gradcam_grid.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*60)
    print("  POST-TRAINING EVALUATION")
    print("="*60)

    # Check models exist
    for p, n in [(CUSTOM_MODEL_PATH, "Custom CNN"),
                 (PRETRAINED_MODEL_PATH, "ResNet50")]:
        if not os.path.exists(p):
            print(f"  ✗ {n} model not found at {p}. Skipping.")
            return

    # Load models
    print("\n[1/5] Loading models...")
    cnn_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    res_model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)

    # Data
    print("[2/5] Loading test data...")
    _, _, test_gen = get_data_generators()

    # Evaluate
    print("[3/5] Running predictions on test set...")
    print("  Custom CNN:")
    y_true, y_pred_cnn, y_probs_cnn = evaluate_model(cnn_model, test_gen, "Custom CNN")
    print("  ResNet50:")
    y_true, y_pred_res, y_probs_res = evaluate_model(res_model, test_gen, "ResNet50")

    # Reports
    print("[4/5] Generating metrics and plots...")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def get_metrics(y_true, y_pred, y_probs):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec  = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        return acc, prec, rec, f1, roc_auc

    cnn_metrics = get_metrics(y_true, y_pred_cnn, y_probs_cnn)
    res_metrics = get_metrics(y_true, y_pred_res, y_probs_res)

    # Confusion matrices
    plot_confusion_matrix(y_true, y_pred_cnn, "Confusion Matrix – Custom CNN",
                          os.path.join(REPORTS_DIR, "cnn_confusion_matrix.png"))
    plot_confusion_matrix(y_true, y_pred_res, "Confusion Matrix – ResNet50",
                          os.path.join(REPORTS_DIR, "resnet50_confusion_matrix.png"))

    # ROC curves
    plot_roc(y_true, y_probs_cnn, "ROC Curve – Custom CNN",
             os.path.join(REPORTS_DIR, "cnn_roc.png"), color='#4facfe')
    plot_roc(y_true, y_probs_res, "ROC Curve – ResNet50",
             os.path.join(REPORTS_DIR, "resnet50_roc.png"), color='#f093fb')

    # Comparison bar chart
    comparison_bar(list(cnn_metrics), list(res_metrics),
                   os.path.join(REPORTS_DIR, "model_comparison.png"))

    # Grad-CAM grids
    print("[5/5] Generating Grad-CAM sample grids...")
    gradcam_sample_grid(cnn_model,  "Custom CNN")
    gradcam_sample_grid(res_model,  "ResNet50")

    # Text summary
    summary_path = os.path.join(REPORTS_DIR, "evaluation_summary.txt")
    labels_names = ['Accuracy','Precision','Recall','F1','AUC']
    with open(summary_path, 'w') as f:
        f.write("PNEUMONIA DETECTION – EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Metric':<15} {'Custom CNN':>12} {'ResNet50':>12}\n")
        f.write("-"*40 + "\n")
        for lbl, c, r in zip(labels_names, cnn_metrics, res_metrics):
            f.write(f"{lbl:<15} {c:>12.4f} {r:>12.4f}\n")
        f.write("\n\nDetailed Classification Reports\n")
        f.write("-"*40 + "\n")
        f.write("Custom CNN:\n")
        f.write(classification_report(y_true, y_pred_cnn,
                target_names=['Normal','Pneumonia']))
        f.write("\nResNet50:\n")
        f.write(classification_report(y_true, y_pred_res,
                target_names=['Normal','Pneumonia']))
    print(f"  Saved: {summary_path}")

    # Print to console
    print("\n" + "="*60)
    print(f"  {'Metric':<15} {'Custom CNN':>12} {'ResNet50':>12}")
    print("  " + "-"*40)
    for lbl, c, r in zip(labels_names, cnn_metrics, res_metrics):
        print(f"  {lbl:<15} {c:>12.4f} {r:>12.4f}")
    print("="*60)
    print(f"\n  All reports saved to: {REPORTS_DIR}")
    print("  Run `python -m streamlit run app.py` to launch the web app.")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()
