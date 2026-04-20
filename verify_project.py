"""
verify_project.py
-----------------
Quick verification script:
 - Loads the trained Custom CNN
 - Runs predictions on 1 PNEUMONIA + 1 NORMAL test image
 - Generates Grad-CAM heatmaps and prints AI interpretation
"""

import os, random
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CUSTOM_MODEL_PATH, IMG_SIZE
from src.gradcam import (get_gradcam_heatmap, overlay_heatmap_on_image,
                          find_last_conv_layer, interpret_heatmap)

def verify():
    print("=" * 60)
    print("  PneumoScan AI - Project Verification")
    print("=" * 60)

    if not os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Model not found at {CUSTOM_MODEL_PATH}. Run main.py first.")
        return

    model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    last_conv = find_last_conv_layer(model)
    print(f"  Model loaded. Last conv layer for Grad-CAM: '{last_conv}'")

    test_p = r"c:\Users\mdkan\OneDrive\Desktop\nndl\chest_xray\test\PNEUMONIA"
    test_n = r"c:\Users\mdkan\OneDrive\Desktop\nndl\chest_xray\test\NORMAL"
    p_imgs = [os.path.join(test_p, f) for f in os.listdir(test_p)
              if f.lower().endswith(('.jpeg','.jpg','.png'))]
    n_imgs = [os.path.join(test_n, f) for f in os.listdir(test_n)
              if f.lower().endswith(('.jpeg','.jpg','.png'))]

    samples = [random.choice(p_imgs), random.choice(n_imgs)]
    labels  = ["PNEUMONIA", "NORMAL"]

    os.makedirs("verification_results", exist_ok=True)

    for i, (img_path, true_label) in enumerate(zip(samples, labels)):
        print(f"\n  Image {i+1}: {os.path.basename(img_path)}")
        print(f"  True Label : {true_label}")

        img_pil   = Image.open(img_path).convert("RGB")
        img_arr   = np.expand_dims(np.array(img_pil.resize(IMG_SIZE)) / 255.0, 0).astype(np.float32)

        pred_raw  = float(model.predict(img_arr, verbose=0)[0][0])
        pred_lbl  = "PNEUMONIA" if pred_raw > 0.5 else "NORMAL"
        conf      = pred_raw if pred_raw > 0.5 else 1 - pred_raw

        print(f"  Prediction : {pred_lbl} ({conf*100:.2f}%)")
        print(f"  Correct    : {'YES' if pred_lbl == true_label else 'NO'}")

        # Grad-CAM
        heatmap         = get_gradcam_heatmap(model, img_arr, last_conv)
        cam_pil, stats  = overlay_heatmap_on_image(img_pil.resize(IMG_SIZE), heatmap)
        analysis        = interpret_heatmap(stats, pred_lbl, conf)

        save_path = os.path.join("verification_results", f"sample_{i+1}_gradcam.jpg")
        cam_pil.save(save_path)
        print(f"  Heatmap    : saved -> {save_path}")
        print(f"  Hot area % : {stats['hot_area_pct']:.1f}%")
        print(f"  Focus      : {analysis['summary']}")

    print("\n" + "=" * 60)
    print("  Verification complete. Check verification_results/ folder.")
    print("=" * 60)

if __name__ == "__main__":
    verify()
