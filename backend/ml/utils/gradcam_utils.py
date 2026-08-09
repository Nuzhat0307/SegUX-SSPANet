"""
GradCAM utilities — wrapper for the pytorch-grad-cam library.
Supports GradCAM, GradCAM++, and EigenGradCAM for multi-method explainability.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional


def compute_gradcam_variants(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    rgb_image: np.ndarray,
) -> List[dict]:
    """
    Compute GradCAM, GradCAM++, and EigenGradCAM visualizations.

    Args:
        model: The trained model
        input_tensor: Preprocessed input tensor (1, C, H, W)
        target_layer: The layer to compute CAMs on
        rgb_image: Original RGB image normalized to [0, 1]

    Returns:
        List of dicts with method name, heatmap, and overlay arrays.
    """
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        raise ImportError(
            "pytorch-grad-cam is required. Install with: pip install grad-cam"
        )

    cam_methods = [
        ("gradcam", GradCAM),
        ("gradcam_plus_plus", GradCAMPlusPlus),
        ("eigengradcam", EigenGradCAM),
    ]

    results = []
    for name, CamClass in cam_methods:
        cam = CamClass(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0]

        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        results.append({
            "method": name,
            "heatmap": grayscale_cam,
            "overlay": visualization,
        })

    return results


def compute_eigengradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """
    EigenGradCAM: Uses PCA of gradients for more stable explanations.
    """
    model.zero_grad()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]

    # Get gradients and activations
    target_layer_output = None
    target_layer_grad = None

    def forward_hook(module, input, output):
        nonlocal target_layer_output
        target_layer_output = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal target_layer_grad
        target_layer_grad = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    score.backward()

    # Eigen decomposition of gradients
    grads = target_layer_grad.squeeze(0)  # (C, H, W)
    activations = target_layer_output.squeeze(0)  # (C, H, W)

    # Reshape for PCA: (H*W, C)
    grads_2d = grads.view(grads.shape[0], -1).T  # (H*W, C)
    U, S, Vh = torch.linalg.svd(grads_2d, full_matrices=False)
    eigen_vector = Vh[0]  # First principal component

    # Weight activations by eigen vector
    weights = eigen_vector.unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1)
    cam = (weights * activations).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    h1.remove()
    h2.remove()
    return cam
