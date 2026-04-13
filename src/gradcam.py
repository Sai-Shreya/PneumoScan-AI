import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def find_last_conv_layer(model):
    """
    Finds the name of the last layer that outputs a spatial (H×W×C) feature map.
    Returns a DIRECT CHILD layer name of `model` — critical for the gradient loop.
    
    For flat Sequential CNNs → returns the last Conv2D layer name.
    For nested models (e.g. ResNet50 backbone inside Sequential) → returns the 
    sub-model's name (e.g. 'resnet50'), because that IS the direct child producing
    the spatial feature map.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Sub-model (e.g. ResNet50 backbone): check if it contains Conv2D layers
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    # Return the PARENT (direct child) name, not the nested sub-layer
                    return layer.name
    return None


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=0):
    """
    Computes Grad-CAM heatmap using GradientTape over a layer-by-layer forward pass.
    Works with both flat Sequential and nested Sequential+Functional models.

    Strategy:
    - Walk through model.layers one-by-one applying each layer.
    - When we hit `last_conv_layer_name`, capture output and watch it with the tape.
    - Compute gradients of the prediction w.r.t. the captured activation.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_out = None

        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                # Capture and register this tensor as a watched leaf
                conv_out = x
                tape.watch(conv_out)

        if conv_out is None:
            raise ValueError(
                f"Layer '{last_conv_layer_name}' not found in model.layers. "
                f"Available: {[l.name for l in model.layers]}"
            )

        preds = x
        # For binary sigmoid output, use index 0
        class_channel = preds[:, pred_index]

    # Compute gradients w.r.t. the conv activation
    grads = tape.gradient(class_channel, conv_out)

    if grads is None:
        raise ValueError("Gradients are None. The conv layer may not be in the computational graph.")

    # Global average pooling of gradients → importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    activation = conv_out[0]
    heatmap = activation @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)
    heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.45):
    """
    Superimposes a Grad-CAM heatmap onto a PIL image.
    Returns: (overlaid PIL.Image, stats_dict)
    """
    # Resize heatmap to match image
    h, w = img_pil.size[1], img_pil.size[0]
    heatmap_uint8 = np.uint8(255 * heatmap)

    jet  = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]       # shape (256, 3)
    jet_heatmap = jet_colors[heatmap_uint8]        # shape (H_hm, W_hm, 3)

    jet_pil = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_pil = jet_pil.resize((w, h), Image.LANCZOS)
    jet_arr = np.array(jet_pil).astype(np.float32)

    orig_arr = np.array(img_pil.convert("RGB")).astype(np.float32)
    blended  = jet_arr * alpha + orig_arr * (1 - alpha)
    blended  = np.clip(blended, 0, 255).astype(np.uint8)

    # Compute stats for analysis
    stats = {
        "mean_activation": float(np.mean(heatmap)),
        "max_activation":  float(np.max(heatmap)),
        "hot_area_pct":    float(np.mean(heatmap > 0.5) * 100),   # % pixels above 50%
        "very_hot_pct":    float(np.mean(heatmap > 0.75) * 100),  # % pixels above 75%
    }

    return Image.fromarray(blended), stats


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """Legacy helper: save heatmap overlay to file."""
    img_pil = Image.open(img_path).convert("RGB")
    overlaid, _ = overlay_heatmap_on_image(img_pil, heatmap, alpha)
    overlaid.save(cam_path)
    return overlaid


def interpret_heatmap(stats, prediction_label, confidence):
    """
    Generate a natural-language interpretation of the Grad-CAM heatmap.
    Returns a multi-paragraph string.
    """
    hot_pct  = stats["hot_area_pct"]
    vhot_pct = stats["very_hot_pct"]
    mean_act = stats["mean_activation"]

    # Focus pattern description
    if vhot_pct > 20:
        focus = "highly concentrated"
        focus_detail = "The model found a strong, localised region of interest within the lung field."
    elif hot_pct > 30:
        focus = "moderately distributed"
        focus_detail = "Activation is spread across a wider area, suggesting diffuse radiological changes."
    else:
        focus = "weakly distributed"
        focus_detail = "The attention is sparse, which can occur with subtle or borderline findings."

    if prediction_label == "PNEUMONIA":
        clinical = (
            f"The highlighted regions ({hot_pct:.1f}% of the image above 50% activation) "
            f"correspond to areas where radiodensity changes are detectable — consistent with "
            f"consolidation, ground-glass opacity, or infiltrates typical of pneumonia. "
            f"Areas marked in red/yellow in the heatmap indicate the lung zones that most "
            f"influenced the diagnosis. The model's confidence is {confidence*100:.1f}%, and "
            f"attention is {focus}. {focus_detail}"
        )
        recommendation = (
            "⚠️ The AI detected features consistent with pneumonia. Clinical correlation "
            "with physical examination, symptoms, and laboratory findings is strongly advised. "
            "A radiologist review is recommended before any treatment decision."
        )
    else:
        clinical = (
            f"The highlighted regions ({hot_pct:.1f}% of the image above 50% activation) "
            f"show areas the model examined to rule out opacification or consolidation. "
            f"The relatively {focus} and low mean activation ({mean_act:.3f}) across the lung "
            f"fields support the absence of significant radiological abnormalities. "
            f"{focus_detail} Model confidence: {confidence*100:.1f}%."
        )
        recommendation = (
            "✅ The AI found no significant indicators of pneumonia. However, this result "
            "should not replace clinical judgment. If symptoms persist, please consult a physician."
        )

    return {
        "summary": f"Grad-CAM focus pattern is **{focus}** — {vhot_pct:.1f}% of pixels show very high activation (>75%).",
        "clinical": clinical,
        "recommendation": recommendation,
    }
