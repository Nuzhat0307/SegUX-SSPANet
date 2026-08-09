"""
Inference service — orchestrates the full SegUX-SSPANet pipeline:

1. Preprocessing
2. Classification (SSPANet + ResNet50)
3. Segmentation (U-Net)
4. Explainability (GradCAM, GradCAM++, EigenGradCAM)
5. Uncertainty estimation (Monte Carlo Dropout)

When the trained PyTorch checkpoint is available, the real model
is used. Otherwise, the service falls back to deterministic mock
inference for development.
"""

import base64
import io
import time
from io import BytesIO
from typing import Dict, Any, List
import numpy as np
from PIL import Image
from loguru import logger

from app.core.config import settings


# ============================================================
# DISPLAY LABELS
# ============================================================

TUMOR_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "no_tumor": "No Tumor",
}


# ============================================================
# INFERENCE SERVICE
# ============================================================

class InferenceService:
    """Manages model loading and inference."""

    def __init__(self):
        self._model = None
        self._segmentor = None
        self._device = "cpu"
        self._loaded = False

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True when the trained checkpoint is loaded."""
        return self._loaded

    # --------------------------------------------------------
    # LOAD TRAINED MODELS
    # --------------------------------------------------------

    def load_models(self):
        """
        Load the trained classifier and segmentation model.

        The checkpoint must contain:
            classifier
            segmentor
        """

        import os
        import torch

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint_path = settings.MODEL_CHECKPOINT_PATH

        logger.info(
            f"Looking for model checkpoint: {checkpoint_path}"
        )

        if not os.path.exists(checkpoint_path):

            logger.error(
                f"MODEL CHECKPOINT NOT FOUND: {checkpoint_path}"
            )

            self._loaded = False
            return

        try:

            from ml.models.segux_sspanet import SegUXSSPANet
            from ml.models.unet import UNet

            # ------------------------------------------------
            # Create models
            # ------------------------------------------------

            self._model = SegUXSSPANet(
                num_classes=settings.NUM_CLASSES,
                backbone="resnet50",
            ).to(self._device)

            self._segmentor = UNet(
                in_channels=1,
                out_channels=1,
            ).to(self._device)

            # ------------------------------------------------
            # Load checkpoint
            # ------------------------------------------------

            logger.info(
                f"Loading checkpoint: {checkpoint_path}"
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self._device,
            )

            # ------------------------------------------------
            # Load classifier
            # ------------------------------------------------

            if isinstance(checkpoint, dict) and "classifier" in checkpoint:

                self._model.load_state_dict(
                    checkpoint["classifier"]
                )

            else:

                self._model.load_state_dict(
                    checkpoint
                )

            # ------------------------------------------------
            # Load segmentor
            # ------------------------------------------------

            if (
                isinstance(checkpoint, dict)
                and "segmentor" in checkpoint
            ):

                self._segmentor.load_state_dict(
                    checkpoint["segmentor"]
                )

                logger.info(
                    "Segmentation model weights loaded."
                )

            else:

                logger.warning(
                    "No 'segmentor' weights found in checkpoint."
                )

            # ------------------------------------------------
            # Evaluation mode
            # ------------------------------------------------

            self._model.eval()
            self._segmentor.eval()

            self._loaded = True

            logger.info(
                "=============================================="
            )
            logger.info(
                "TRAINED MODEL LOADED SUCCESSFULLY"
            )
            logger.info(
                f"Checkpoint: {checkpoint_path}"
            )
            logger.info(
                f"Device: {self._device}"
            )
            logger.info(
                "=============================================="
            )

        except Exception as e:

            self._loaded = False

            logger.exception(
                f"FAILED TO LOAD TRAINED MODEL: {e}"
            )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    async def predict(
        self,
        image_base64: str,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Run inference on a base64-encoded MRI image.
        """

        start_time = time.time()

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        # Decode Base64
        image_bytes = base64.b64decode(image_base64)

        # Open image and convert grayscale MRI to 3-channel RGB
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # ----------------------------------------------------
        # REAL MODEL
        # ----------------------------------------------------

        if (
            self._loaded
            and self._model is not None
            and self._segmentor is not None
        ):

            logger.info(
                f"Running REAL model inference for patient: "
                f"{patient_id}"
            )

            result = self._real_inference(
                image
            )

        # ----------------------------------------------------
        # MOCK MODEL
        # ----------------------------------------------------

        else:

            logger.warning(
                "Trained model is not loaded. "
                "Using mock inference."
            )

            result = self._mock_inference(
                image
            )

        # ----------------------------------------------------
        # Common response fields
        # ----------------------------------------------------

        result["inference_time_ms"] = int(
            (time.time() - start_time) * 1000
        )

        result["model_version"] = (
            settings.MODEL_VERSION
        )

        result["patient_id"] = patient_id

        return result

    # --------------------------------------------------------
    # BASE64 DECODER
    # --------------------------------------------------------

    def _decode_base64_image(
        self,
        b64_str: str,
    ) -> bytes:

        """Decode a base64 data URL or raw base64 string."""

        if "," in b64_str:
            b64_str = b64_str.split(
                ",",
                1,
            )[1]

        return base64.b64decode(
            b64_str
        )

    # ========================================================
    # REAL INFERENCE
    # ========================================================

    def _real_inference(
            self,
            image: Image.Image,
    ) -> Dict[str, Any]:

        """Run actual trained PyTorch inference."""

        import torch

        # ====================================================
        # CLASSIFICATION PREPROCESSING
        # ====================================================

        # Make absolutely sure the MRI is RGB.
        image_rgb = image.convert("RGB")

        img_array = np.array(
            image_rgb.resize(
                (
                    settings.IMAGE_SIZE,
                    settings.IMAGE_SIZE,
                )
            ),
            dtype=np.float32,
        )

        # HWC -> CHW
        # (224, 224, 3) -> (3, 224, 224)
        img_array = np.transpose(
            img_array,
            (2, 0, 1),
        )

        # Add batch dimension
        # (3, 224, 224) -> (1, 3, 224, 224)
        img_tensor = (
                torch.from_numpy(img_array)
                .float()
                .unsqueeze(0)
                .to(self._device)
                / 255.0
        )

        logger.info(
            f"Classification input shape: {tuple(img_tensor.shape)}"
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        with torch.no_grad():
            logits = self._model(
                img_tensor
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()[0]
            )

        # ====================================================
        # MC DROPOUT UNCERTAINTY
        # ====================================================

        uncertainty = (
            self._mc_dropout_uncertainty(
                img_tensor
            )
        )

        # ====================================================
        # SEGMENTATION PREPROCESSING
        # ====================================================

        # U-Net was created with in_channels=1,
        # so segmentation should receive grayscale.
        gray_image = image_rgb.convert("L")

        seg_array = np.array(
            gray_image.resize(
                (
                    settings.SEGMENTATION_SIZE,
                    settings.SEGMENTATION_SIZE,
                )
            ),
            dtype=np.float32,
        )

        # H,W -> 1,H,W -> 1,1,H,W
        seg_input = (
                torch.from_numpy(seg_array)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self._device)
                / 255.0
        )

        logger.info(
            f"Segmentation input shape: {tuple(seg_input.shape)}"
        )

        # ====================================================
        # SEGMENTATION
        # ====================================================

        with torch.no_grad():
            seg_output = self._segmentor(
                seg_input
            )

            seg_mask = (
                torch.sigmoid(
                    seg_output
                )
                .cpu()
                .numpy()[0, 0]
            )

        # ====================================================
        # GRADCAM
        # ====================================================

        gradcam_results = (
            self._compute_gradcam(
                img_tensor,
                image_rgb,
            )
        )

        # ====================================================
        # CLASSIFICATION RESULT
        # ====================================================

        predicted_idx = int(
            np.argmax(probs)
        )

        predicted_class = (
            settings.TUMOR_CLASSES[
                predicted_idx
            ]
        )

        probabilities = []

        for i, cls in enumerate(
                settings.TUMOR_CLASSES
        ):
            probabilities.append(
                {
                    "label": cls,

                    "display_name":
                        TUMOR_DISPLAY.get(
                            cls,
                            cls,
                        ),

                    "probability":
                        float(probs[i]),
                }
            )

        probabilities.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        # ====================================================
        # SEGMENTATION RESULT
        # ====================================================

        segmentation = (
            self._build_segmentation_result(
                seg_mask,
                image_rgb,
            )
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {
            "predicted_class":
                predicted_class,

            "predicted_class_display":
                TUMOR_DISPLAY.get(
                    predicted_class,
                    predicted_class,
                ),

            "probabilities":
                probabilities,

            "uncertainty":
                uncertainty,

            "segmentation":
                segmentation,

            "gradcam_results":
                gradcam_results,
        }

    # ========================================================
    # MC DROPOUT
    # ========================================================

    def _mc_dropout_uncertainty(
            self,
            img_tensor,
    ) -> Dict[str, Any]:

        """Monte Carlo Dropout uncertainty estimation."""

        import torch
        import torch.nn as nn

        # ----------------------------------------------------
        # Keep the complete model in evaluation mode first.
        # This prevents BatchNorm from trying to calculate
        # statistics from a single image.
        # ----------------------------------------------------

        self._model.eval()

        # ----------------------------------------------------
        # Enable ONLY Dropout layers.
        # Do NOT call self._model.train(), because that would
        # also put BatchNorm layers into training mode.
        # ----------------------------------------------------

        for module in self._model.modules():

            if isinstance(
                    module,
                    (
                            nn.Dropout,
                            nn.Dropout1d,
                            nn.Dropout2d,
                            nn.Dropout3d,
                            nn.AlphaDropout,
                    ),
            ):
                module.train()

        # ----------------------------------------------------
        # Monte Carlo forward passes
        # ----------------------------------------------------

        all_probs = []

        with torch.no_grad():

            for _ in range(
                    settings.MC_DROPOUT_SAMPLES
            ):
                logits = self._model(
                    img_tensor
                )

                probs = (
                    torch.softmax(
                        logits,
                        dim=1,
                    )
                    .cpu()
                    .numpy()[0]
                )

                all_probs.append(probs)

        # ----------------------------------------------------
        # IMPORTANT:
        # Return the model completely to evaluation mode.
        # ----------------------------------------------------

        self._model.eval()

        all_probs = np.array(
            all_probs
        )

        mean_probs = (
            all_probs.mean(axis=0)
        )

        # ----------------------------------------------------
        # Predictive entropy
        # ----------------------------------------------------

        pred_entropy = -np.sum(
            mean_probs
            * np.log2(
                mean_probs + 1e-10
            )
        )

        # ----------------------------------------------------
        # Expected entropy
        # ----------------------------------------------------

        expected_entropy = np.mean(
            [
                -np.sum(
                    p
                    * np.log2(
                        p + 1e-10
                    )
                )
                for p in all_probs
            ]
        )

        # ----------------------------------------------------
        # Mutual information
        # ----------------------------------------------------

        mutual_info = (
                pred_entropy
                - expected_entropy
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        max_entropy = np.log2(
            settings.NUM_CLASSES
        )

        confidence = (
                1
                - pred_entropy / max_entropy
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # Final uncertainty result
        # ----------------------------------------------------

        return {
            "method":
                "monte_carlo_dropout",

            "num_samples":
                settings.MC_DROPOUT_SAMPLES,

            "predictive_entropy":
                float(pred_entropy),

            "mutual_information":
                float(
                    max(
                        0,
                        mutual_info,
                    )
                ),

            "confidence":
                confidence,

            "is_uncertain":
                bool(
                    confidence
                    < settings.UNCERTAINTY_THRESHOLD
                    or mutual_info > 0.3
                ),
        }

    # ========================================================
    # GRADCAM
    # ========================================================

    def _compute_gradcam(
        self,
        img_tensor,
        image: Image.Image,
    ) -> List[Dict[str, str]]:

        """
        Compute GradCAM, GradCAM++, and EigenGradCAM.

        If GradCAM cannot be computed, a fallback visualization
        is returned.
        """

        try:

            from pytorch_grad_cam import (
                GradCAM,
                GradCAMPlusPlus,
                EigenGradCAM,
            )

            from pytorch_grad_cam.utils.image import (
                show_cam_on_image,
            )

            target_layer = (
                self._model.get_target_layer()
            )

            methods = [
                (
                    "gradcam",
                    GradCAM,
                ),
                (
                    "gradcam_plus_plus",
                    GradCAMPlusPlus,
                ),
                (
                    "eigengradcam",
                    EigenGradCAM,
                ),
            ]

            results = []

            rgb_img = np.array(
                image.resize(
                    (
                        settings.IMAGE_SIZE,
                        settings.IMAGE_SIZE,
                    )
                ).convert("RGB")
            ) / 255.0

            rgb_img = np.float32(
                rgb_img
            )

            for name, CamClass in methods:

                cam = CamClass(
                    model=self._model,
                    target_layers=[
                        target_layer
                    ],
                )

                grayscale_cam = cam(
                    input_tensor=img_tensor
                )

                grayscale_cam = (
                    grayscale_cam[0]
                )

                visualization = (
                    show_cam_on_image(
                        rgb_img,
                        grayscale_cam,
                        use_rgb=True,
                    )
                )

                heatmap_b64 = (
                    self._encode_image_array(
                        visualization
                    )
                )

                overlay_b64 = (
                    self._encode_image_array(
                        visualization
                    )
                )

                results.append(
                    {
                        "method": name,
                        "heatmap_base64":
                            heatmap_b64,
                        "overlay_base64":
                            overlay_b64,
                    }
                )

            return results

        except Exception as e:

            logger.warning(
                f"GradCAM computation failed: {e}"
            )

            return self._mock_gradcam(
                image
            )

    # ========================================================
    # SEGMENTATION RESULT
    # ========================================================

    def _build_segmentation_result(
        self,
        mask: np.ndarray,
        image: Image.Image,
    ) -> Dict[str, Any]:

        import cv2

        binary_mask = (
            mask > 0.5
        ).astype(
            np.uint8
        ) * 255

        tumor_pixels = int(
            np.sum(
                binary_mask > 0
            )
        )

        total_pixels = (
            binary_mask.shape[0]
            * binary_mask.shape[1]
        )

        tumor_percentage = (
            tumor_pixels
            / total_pixels
        ) * 100

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        bbox = None

        if contours:

            x, y, w, h = cv2.boundingRect(
                max(
                    contours,
                    key=cv2.contourArea,
                )
            )

            bbox = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            }

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        orig = np.array(
            image.resize(
                (
                    settings.SEGMENTATION_SIZE,
                    settings.SEGMENTATION_SIZE,
                )
            ).convert("RGB")
        )

        overlay = orig.copy()

        overlay[
            binary_mask > 0
        ] = [220, 38, 38]

        overlay = cv2.addWeighted(
            orig,
            0.55,
            overlay,
            0.45,
            0,
        )

        # ----------------------------------------------------
        # NOTE:
        # Dice cannot be calculated without ground truth.
        # Do NOT present a random Dice value as real accuracy.
        # ----------------------------------------------------

        return {
            "mask_base64":
                self._encode_image_array(
                    binary_mask
                ),

            "overlay_base64":
                self._encode_image_array(
                    overlay
                ),

            "dice_score":
                None,

            "tumor_area_pixels":
                tumor_pixels,

            "tumor_area_percentage":
                float(tumor_percentage),

            "bounding_box":
                bbox,
        }

    # ========================================================
    # MOCK INFERENCE
    # ========================================================

    def _mock_inference(
        self,
        image: Image.Image,
    ) -> Dict[str, Any]:

        """
        Deterministic mock inference.

        This should only be reached if the trained checkpoint
        cannot be loaded.
        """

        import cv2

        img_arr = np.array(
            image.resize(
                (
                    224,
                    224,
                )
            ),
            dtype=np.float32,
        )

        brightness = img_arr.mean()
        contrast = img_arr.std()

        seed = abs(
            int(
                img_arr.sum()
            )
        ) % (2**31)

        rng = np.random.RandomState(
            seed
        )

        logits = np.array(
            [
                rng.randn()
                + (
                    1.5
                    if brightness < 100
                    else 0.3
                ),

                rng.randn()
                + (
                    1.2
                    if contrast > 55
                    else 0.2
                ),

                rng.randn()
                + (
                    1.3
                    if brightness > 110
                    else 0.2
                ),

                rng.randn()
                + (
                    2.0
                    if contrast < 40
                    else -0.5
                ),
            ]
        )

        probs = np.exp(
            logits - logits.max()
        )

        probs /= probs.sum()

        predicted_idx = int(
            probs.argmax()
        )

        predicted_class = (
            settings.TUMOR_CLASSES[
                predicted_idx
            ]
        )

        probabilities = [
            {
                "label": cls,
                "display_name":
                    TUMOR_DISPLAY.get(
                        cls,
                        cls,
                    ),
                "probability":
                    float(probs[i]),
            }
            for i, cls in enumerate(
                settings.TUMOR_CLASSES
            )
        ]

        probabilities.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Mock uncertainty
        # ----------------------------------------------------

        all_probs = []

        for _ in range(
            settings.MC_DROPOUT_SAMPLES
        ):

            noisy = (
                probs
                + rng.randn(4) * 0.05
            )

            noisy = np.exp(
                noisy - noisy.max()
            )

            noisy /= noisy.sum()

            all_probs.append(
                noisy
            )

        all_probs = np.array(
            all_probs
        )

        mean_probs = (
            all_probs.mean(axis=0)
        )

        pred_entropy = -np.sum(
            mean_probs
            * np.log2(
                mean_probs + 1e-10
            )
        )

        expected_entropy = np.mean(
            [
                -np.sum(
                    p
                    * np.log2(
                        p + 1e-10
                    )
                )
                for p in all_probs
            ]
        )

        mutual_info = max(
            0,
            pred_entropy
            - expected_entropy,
        )

        confidence = (
            1
            - pred_entropy
            / np.log2(4)
        )

        uncertainty = {
            "method":
                "monte_carlo_dropout",

            "num_samples":
                settings.MC_DROPOUT_SAMPLES,

            "predictive_entropy":
                float(pred_entropy),

            "mutual_information":
                float(mutual_info),

            "confidence":
                float(confidence),

            "is_uncertain":
                bool(
                    confidence < 0.75
                    or mutual_info > 0.3
                ),
        }

        segmentation = (
            self._mock_segmentation(
                image,
                predicted_class,
                rng,
            )
        )

        gradcam_results = (
            self._mock_gradcam(
                image,
                segmentation,
            )
        )

        return {
            "predicted_class":
                predicted_class,

            "predicted_class_display":
                TUMOR_DISPLAY.get(
                    predicted_class,
                    predicted_class,
                ),

            "probabilities":
                probabilities,

            "uncertainty":
                uncertainty,

            "segmentation":
                segmentation,

            "gradcam_results":
                gradcam_results,
        }

    # ========================================================
    # MOCK SEGMENTATION
    # ========================================================

    def _mock_segmentation(
        self,
        image: Image.Image,
        predicted_class: str,
        rng,
    ) -> Dict[str, Any]:

        import cv2

        size = (
            settings.SEGMENTATION_SIZE
        )

        img = np.array(
            image.resize(
                (
                    size,
                    size,
                )
            ).convert("RGB")
        )

        if predicted_class == "no_tumor":

            blank = np.zeros(
                (
                    size,
                    size,
                ),
                dtype=np.uint8,
            )

            return {
                "mask_base64":
                    self._encode_image_array(
                        blank
                    ),

                "overlay_base64":
                    self._encode_image_array(
                        img
                    ),

                "dice_score":
                    None,

                "tumor_area_pixels":
                    0,

                "tumor_area_percentage":
                    0.0,

                "bounding_box":
                    None,
            }

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY,
        )

        blur = cv2.GaussianBlur(
            gray,
            (31, 31),
            0,
        )

        _, _, _, max_loc = (
            cv2.minMaxLoc(
                blur
            )
        )

        cx, cy = max_loc

        rx = (
            25
            + rng.randint(20)
        )

        ry = (
            20
            + rng.randint(20)
        )

        mask = np.zeros(
            (
                size,
                size,
            ),
            dtype=np.uint8,
        )

        cv2.ellipse(
            mask,
            (cx, cy),
            (rx, ry),
            0,
            0,
            360,
            255,
            -1,
        )

        tumor_pixels = int(
            np.sum(
                mask > 0
            )
        )

        overlay = img.copy()

        overlay[
            mask > 0
        ] = [220, 38, 38]

        overlay = cv2.addWeighted(
            img,
            0.55,
            overlay,
            0.45,
            0,
        )

        cv2.ellipse(
            overlay,
            (cx, cy),
            (rx, ry),
            0,
            0,
            360,
            (220, 38, 38),
            2,
        )

        return {
            "mask_base64":
                self._encode_image_array(
                    mask
                ),

            "overlay_base64":
                self._encode_image_array(
                    overlay
                ),

            "dice_score":
                None,

            "tumor_area_pixels":
                tumor_pixels,

            "tumor_area_percentage":
                float(
                    tumor_pixels
                    / (size * size)
                    * 100
                ),

            "bounding_box": {
                "x":
                    int(cx - rx),

                "y":
                    int(cy - ry),

                "width":
                    int(rx * 2),

                "height":
                    int(ry * 2),
            },
        }

    # ========================================================
    # MOCK GRADCAM
    # ========================================================

    def _mock_gradcam(
        self,
        image: Image.Image,
        segmentation: Dict = None,
    ) -> List[Dict[str, str]]:

        import cv2

        size = 256

        img = np.array(
            image.resize(
                (
                    size,
                    size,
                )
            ).convert("RGB")
        )

        bbox = (
            segmentation.get(
                "bounding_box"
            )
            if segmentation
            else None
        )

        if bbox:

            cx = (
                bbox["x"]
                + bbox["width"] // 2
            )

            cy = (
                bbox["y"]
                + bbox["height"] // 2
            )

            radius = (
                max(
                    bbox["width"],
                    bbox["height"],
                )
                // 2
                + 15
            )

        else:

            cx = size // 2
            cy = size // 2
            radius = 50

        results = []

        for method, spread in [
            ("gradcam", 1.5),
            ("gradcam_plus_plus", 2.0),
            ("eigengradcam", 1.2),
        ]:

            heatmap = np.zeros(
                (
                    size,
                    size,
                    3,
                ),
                dtype=np.uint8,
            )

            for y in range(size):

                for x in range(size):

                    dist = (
                        np.sqrt(
                            (x - cx) ** 2
                            + (y - cy) ** 2
                        )
                        / radius
                    )

                    val = np.clip(
                        np.exp(
                            -dist
                            * dist
                            * spread
                        ),
                        0,
                        1,
                    )

                    if val < 0.25:

                        heatmap[y, x] = [
                            0,
                            int(
                                val
                                * 4
                                * 255
                            ),
                            255,
                        ]

                    elif val < 0.5:

                        heatmap[y, x] = [
                            0,
                            255,
                            int(
                                (
                                    0.5
                                    - val
                                )
                                * 4
                                * 255
                            ),
                        ]

                    elif val < 0.75:

                        heatmap[y, x] = [
                            int(
                                (
                                    val
                                    - 0.5
                                )
                                * 4
                                * 255
                            ),
                            255,
                            0,
                        ]

                    else:

                        heatmap[y, x] = [
                            255,
                            int(
                                (
                                    1
                                    - val
                                )
                                * 4
                                * 255
                            ),
                            0,
                        ]

            overlay = cv2.addWeighted(
                img,
                0.5,
                heatmap,
                0.5,
                0,
            )

            results.append(
                {
                    "method":
                        method,

                    "heatmap_base64":
                        self._encode_image_array(
                            heatmap
                        ),

                    "overlay_base64":
                        self._encode_image_array(
                            overlay
                        ),
                }
            )

        return results

    # ========================================================
    # IMAGE ENCODING
    # ========================================================

    def _encode_image_array(
        self,
        arr: np.ndarray,
    ) -> str:

        """Encode numpy image as base64 PNG."""

        import cv2

        if arr.ndim == 2:

            arr = cv2.cvtColor(
                arr,
                cv2.COLOR_GRAY2RGB,
            )

        elif (
            arr.ndim == 3
            and arr.shape[2] == 4
        ):

            arr = cv2.cvtColor(
                arr,
                cv2.COLOR_RGBA2RGB,
            )

        success, buffer = cv2.imencode(
            ".png",
            arr,
        )

        if not success:

            raise RuntimeError(
                "Failed to encode image"
            )

        b64 = base64.b64encode(
            buffer
        ).decode(
            "utf-8"
        )

        return (
            "data:image/png;base64,"
            + b64
        )


# ============================================================
# GLOBAL INFERENCE SERVICE
# ============================================================

inference_service = InferenceService()