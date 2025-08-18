## 3D-HCCT: A Hybrid CNN–ViT Architecture

**3D-HCCT** is a hybrid deep learning architecture. The model operates in two main phases:

### 🔹 CNN Phase

This phase processes the input volume (e.g., of shape `(1, 1, 91, 109, 91)`) and generates a **feature map**.

* The output feature map from the last CNN layer has a shape of `[1, 512, 2, 3, 2]`.
* These features are then flattened and fed into the ViT phase.

This phase also supports **Grad-CAM** visualization by returning:

* `feature_map`: Output of the last CNN layer (e.g., `[1, 512, 2, 3, 2]`)
* `gradients`: Corresponding gradients with respect to the `feature_map`

### 🔹 ViT Phase

The Vision Transformer expects a sequence of patch embeddings. Therefore, the CNN feature map is reshaped to `[1, 512, 12]`, where:

* `512` represents the number of patch tokens.
* `12` corresponds to the flattened embedding from the spatial dimensions `(2×3×2)`.

The ViT consists of **3 Transformer blocks**, each with **8 attention heads**.

This phase returns:

* `age`: The final regression output.
* `attention_maps`: A list of 3 attention probability maps (one from each block) used for visualization and interpretability.



### Model Inputs

The model accepts two optional flags during forward pass:

* `output_attentions` (`bool`):
  If `True`, returns attention maps from each ViT block.

* `return_cam` (`bool`):
  If `True`, returns the feature map and gradients needed for Grad-CAM analysis.