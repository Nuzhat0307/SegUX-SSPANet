"""
Inference service for SegUX-SSPANet.

Pipeline:

1. Decode MRI
2. Classification preprocessing
3. U-Net segmentation
4. Segmentation-guided SegUX-SSPANet classification
5. Monte Carlo Dropout uncertainty
6. Segmentation visualization
7. GradCAM / GradCAM++ / EigenGradCAM
8. Return API-safe result

IMPORTANT:
The classifier was trained with segmentation guidance, therefore
inference MUST also pass the U-Net segmentation guidance to the
classifier.

Class mapping:
    0 = glioma
    1 = meningioma
    2 = pituitary
    3 = no_tumor

Dice score:
    Dice requires a ground-truth segmentation mask.
    A newly uploaded MRI has no ground truth, so Dice is returned
    as None rather than inventing an accuracy value.
"""

import base64
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

    # ========================================================
    # MODEL STATUS
    # ========================================================

    def is_loaded(self) -> bool:
        return self._loaded

    # ========================================================
    # LOAD TRAINED MODELS
    # ========================================================

    def load_models(self):
        """
        Load the trained SegUX-SSPANet classifier and U-Net.

        Expected checkpoint:

            {
                "classifier": ...,
                "segmentor": ...
            }
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
            # Create classifier
            # ------------------------------------------------

            self._model = SegUXSSPANet(
                num_classes=settings.NUM_CLASSES,
                backbone="resnet50",
            ).to(self._device)

            # ------------------------------------------------
            # Create segmentation model
            # ------------------------------------------------

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
            # Classifier weights
            # ------------------------------------------------

            if (
                isinstance(checkpoint, dict)
                and "classifier" in checkpoint
            ):
                self._model.load_state_dict(
                    checkpoint["classifier"]
                )
            else:
                self._model.load_state_dict(
                    checkpoint
                )

            # ------------------------------------------------
            # Segmentor weights
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

    # ========================================================
    # PREDICT
    # ========================================================

    async def predict(
        self,
        image_base64: str,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Run inference on a base64 encoded MRI image.
        """

        start_time = time.time()

        # ----------------------------------------------------
        # Decode base64
        # ----------------------------------------------------

        if "," in image_base64:
            image_base64 = image_base64.split(
                ",",
                1,
            )[1]

        image_bytes = base64.b64decode(
            image_base64
        )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

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

    # ========================================================
    # BASE64 DECODER
    # ========================================================

    def _decode_base64_image(
        self,
        b64_str: str,
    ) -> bytes:

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

        """
        Real SegUX-SSPANet inference.

        IMPORTANT:

        Classification was trained using segmentation guidance.

        Therefore:

            MRI
             ↓
            grayscale
             ↓
            U-Net
             ↓
            segmentation guidance
             ↓
            SegUX-SSPANet(image, guidance)

        must be preserved.
        """

        import torch
        import torch.nn.functional as F

        # ====================================================
        # 1. CLASSIFICATION PREPROCESSING
        # ====================================================

        # FigshareDataset does:
        #
        #   RGB -> grayscale
        #   resize 224x224
        #   normalize /255
        #   grayscale -> 3 channels
        #
        # Reproduce exactly.

        gray_image = image.convert("L")

        classification_image = gray_image.resize(
            (
                settings.IMAGE_SIZE,
                settings.IMAGE_SIZE,
            )
        )

        img_array = np.array(
            classification_image,
            dtype=np.float32,
        ) / 255.0

        # H,W -> 1,H,W
        img_tensor = (
            torch.from_numpy(
                img_array
            )
            .unsqueeze(0)
        )

        # 1,H,W -> 3,H,W
        img_tensor = img_tensor.repeat(
            3,
            1,
            1,
        )

        # 3,H,W -> 1,3,H,W
        img_tensor = (
            img_tensor
            .unsqueeze(0)
            .to(self._device)
        )

        logger.info(
            f"Classification input shape: "
            f"{tuple(img_tensor.shape)}"
        )

        # ====================================================
        # 2. U-NET SEGMENTATION
        # ====================================================

        # Convert the same grayscale classification image
        # to U-Net input size.

        grayscale = img_tensor.mean(
            dim=1,
            keepdim=True,
        )

        segmentation_input = F.interpolate(
            grayscale,
            size=(
                settings.SEGMENTATION_SIZE,
                settings.SEGMENTATION_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        logger.info(
            f"Segmentation input shape: "
            f"{tuple(segmentation_input.shape)}"
        )

        self._segmentor.eval()

        with torch.no_grad():

            segmentation_logits = (
                self._segmentor(
                    segmentation_input
                )
            )

            full_seg_mask_tensor = (
                torch.sigmoid(
                    segmentation_logits
                )
            )

        logger.info(
            f"Segmentation output shape: "
            f"{tuple(full_seg_mask_tensor.shape)}"
        )

        # ====================================================
        # 3. SEGMENTATION GUIDANCE
        # ====================================================

        # Resize U-Net output from 256x256 to
        # classifier input size 224x224.

        segmentation_guidance = F.interpolate(
            full_seg_mask_tensor,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        segmentation_guidance = (
            segmentation_guidance.clamp(
                0.0,
                1.0,
            )
        )

        logger.info(
            f"Segmentation guidance shape: "
            f"{tuple(segmentation_guidance.shape)}"
        )

        logger.info(
            "Segmentation guidance range: "
            f"min={segmentation_guidance.min().item():.4f}, "
            f"max={segmentation_guidance.max().item():.4f}, "
            f"mean={segmentation_guidance.mean().item():.4f}"
        )

        # ====================================================
        # 4. CLASSIFICATION
        # ====================================================

        self._model.eval()

        with torch.no_grad():

            logits = self._model(
                img_tensor,
                segmentation_guidance,
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()[0]
            )

        # Safety check
        if len(probs) != len(
            settings.TUMOR_CLASSES
        ):
            raise RuntimeError(
                "Number of model outputs does not match "
                "settings.TUMOR_CLASSES"
            )

        logger.info(
            "Classification probabilities: "
            + str(
                {
                    cls: round(
                        float(probs[i]),
                        6,
                    )
                    for i, cls in enumerate(
                        settings.TUMOR_CLASSES
                    )
                }
            )
        )

        # ====================================================
        # 5. MC DROPOUT UNCERTAINTY
        # ====================================================

        uncertainty = (
            self._mc_dropout_uncertainty(
                img_tensor,
                segmentation_guidance,
            )
        )

        # ====================================================
        # 6. GRADCAM
        # ====================================================

        gradcam_results = (
            self._compute_gradcam(
                img_tensor,
                segmentation_guidance,
                image,
            )
        )

        # ====================================================
        # 7. CLASSIFICATION RESULT
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
        # 8. SEGMENTATION RESULT
        # ====================================================

        full_seg_mask = (
            full_seg_mask_tensor
            .detach()
            .cpu()
            .numpy()[0, 0]
        )

        segmentation = (
            self._build_segmentation_result(
                full_seg_mask,
                image,
            )
        )

        # ====================================================
        # 9. FINAL RESULT
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
    # MC DROPOUT UNCERTAINTY
    # ========================================================

    def _mc_dropout_uncertainty(
        self,
        img_tensor,
        segmentation_guidance,
    ) -> Dict[str, Any]:

        """
        Monte Carlo Dropout uncertainty.

        IMPORTANT:

        The returned `confidence` is the actual probability
        of the predicted class.

        Example:

            [0.02, 0.91, 0.04, 0.03]

        confidence = 0.91

        We do NOT use entropy-derived certainty as the
        displayed confidence because that can produce values
        such as 5.7% even when the predicted class probability
        is 33.3%.
        """

        import torch
        import torch.nn as nn

        # ====================================================
        # Keep BatchNorm in evaluation mode
        # ====================================================

        self._model.eval()

        # ====================================================
        # Enable ONLY dropout layers
        # ====================================================

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

        # ====================================================
        # Monte Carlo samples
        # ====================================================

        all_probs = []

        with torch.no_grad():

            for _ in range(
                settings.MC_DROPOUT_SAMPLES
            ):

                logits = self._model(
                    img_tensor,
                    segmentation_guidance,
                )

                probs = (
                    torch.softmax(
                        logits,
                        dim=1,
                    )
                    .cpu()
                    .numpy()[0]
                )

                all_probs.append(
                    probs
                )

        # ====================================================
        # Restore evaluation mode
        # ====================================================

        self._model.eval()

        all_probs = np.asarray(
            all_probs,
            dtype=np.float64,
        )

        mean_probs = all_probs.mean(
            axis=0
        )

        # Numerical safety
        mean_probs = np.clip(
            mean_probs,
            1e-10,
            1.0,
        )

        mean_probs = (
            mean_probs
            / mean_probs.sum()
        )

        # ====================================================
        # PREDICTED CLASS PROBABILITY
        # ====================================================

        top_probability = float(
            np.max(mean_probs)
        )

        predicted_idx = int(
            np.argmax(mean_probs)
        )

        predicted_class = (
            settings.TUMOR_CLASSES[
                predicted_idx
            ]
        )

        # ====================================================
        # PREDICTIVE ENTROPY
        # ====================================================

        pred_entropy = float(
            -np.sum(
                mean_probs
                * np.log2(
                    mean_probs
                )
            )
        )

        # ====================================================
        # EXPECTED ENTROPY
        # ====================================================

        expected_entropy = float(
            np.mean(
                [
                    -np.sum(
                        p
                        * np.log2(
                            np.clip(
                                p,
                                1e-10,
                                1.0,
                            )
                        )
                    )
                    for p in all_probs
                ]
            )
        )

        # ====================================================
        # MUTUAL INFORMATION
        # ====================================================

        mutual_info = max(
            0.0,
            float(
                pred_entropy
                - expected_entropy
            ),
        )

        # ====================================================
        # ENTROPY-BASED CERTAINTY
        # ====================================================

        # Keep this as an additional diagnostic metric.
        # It is NOT exposed as "confidence".

        max_entropy = float(
            np.log2(
                settings.NUM_CLASSES
            )
        )

        entropy_certainty = (
            1.0
            - (
                pred_entropy
                / max_entropy
            )
        )

        entropy_certainty = float(
            np.clip(
                entropy_certainty,
                0.0,
                1.0,
            )
        )

        # ====================================================
        # UNCERTAINTY DECISION
        # ====================================================

        # Primary criterion:
        # actual predicted-class probability.

        confidence_threshold = float(
            getattr(
                settings,
                "UNCERTAINTY_THRESHOLD",
                0.75,
            )
        )

        # Secondary criterion:
        # genuinely high epistemic uncertainty.

        mutual_information_threshold = 0.30

        low_class_confidence = (
            top_probability
            < confidence_threshold
        )

        high_epistemic_uncertainty = (
            mutual_info
            > mutual_information_threshold
        )

        is_uncertain = bool(
            low_class_confidence
            or high_epistemic_uncertainty
        )

        # ====================================================
        # LOGGING
        # ====================================================

        logger.info(
            "MC Dropout uncertainty: "
            f"predicted_class={predicted_class}, "
            f"top_probability={top_probability:.4f}, "
            f"entropy_certainty={entropy_certainty:.4f}, "
            f"predictive_entropy={pred_entropy:.4f}, "
            f"mutual_information={mutual_info:.4f}, "
            f"low_class_confidence={low_class_confidence}, "
            f"high_epistemic_uncertainty="
            f"{high_epistemic_uncertainty}, "
            f"is_uncertain={is_uncertain}"
        )

        # ====================================================
        # RESULT
        # ====================================================

        return {

            "method":
                "monte_carlo_dropout",

            "num_samples":
                settings.MC_DROPOUT_SAMPLES,

            "predictive_entropy":
                float(pred_entropy),

            "mutual_information":
                float(mutual_info),

            # IMPORTANT:
            # This is the actual predicted-class probability.
            "confidence":
                top_probability,

            # Explicit duplicate field for API consumers.
            "top_probability":
                top_probability,

            # Additional diagnostic metric.
            "entropy_certainty":
                entropy_certainty,

            "predicted_class":
                predicted_class,

            "is_uncertain":
                is_uncertain,

            "uncertainty_reason":
                (
                    "low_class_probability"
                    if low_class_confidence
                    else
                    "high_epistemic_uncertainty"
                    if high_epistemic_uncertainty
                    else
                    "none"
                ),
        }

    # ========================================================
    # GRADCAM
    # ========================================================

    def _compute_gradcam(
        self,
        img_tensor,
        segmentation_guidance,
        image: Image.Image,
    ) -> List[Dict[str, str]]:

        """
        Compute GradCAM, GradCAM++, and EigenGradCAM.

        The same segmentation guidance used for classification
        is supplied to the classifier wrapper.
        """

        try:

            import torch
            import torch.nn as nn

            from pytorch_grad_cam import (
                GradCAM,
                GradCAMPlusPlus,
                EigenGradCAM,
            )

            from pytorch_grad_cam.utils.image import (
                show_cam_on_image,
            )

            # ------------------------------------------------
            # Guided classifier wrapper
            # ------------------------------------------------

            class GuidedClassifier(
                nn.Module
            ):

                def __init__(
                    self,
                    model,
                    guidance,
                ):
                    super().__init__()

                    self.model = model
                    self.guidance = guidance

                def forward(self, x):

                    return self.model(
                        x,
                        self.guidance,
                    )

            guided_model = GuidedClassifier(
                self._model,
                segmentation_guidance,
            ).to(self._device)

            guided_model.eval()

            # ------------------------------------------------
            # Target layer
            # ------------------------------------------------

            target_layer = (
                self._model.get_target_layer()
            )

            # ------------------------------------------------
            # Display image
            # ------------------------------------------------

            rgb_img = np.array(
                image.resize(
                    (
                        settings.IMAGE_SIZE,
                        settings.IMAGE_SIZE,
                    )
                ).convert("RGB")
            ).astype(
                np.float32
            ) / 255.0

            # ------------------------------------------------
            # CAM methods
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Generate CAMs
            # ------------------------------------------------

            for name, CamClass in methods:

                try:

                    cam = CamClass(
                        model=guided_model,
                        target_layers=[
                            target_layer
                        ],
                    )

                    grayscale_cam = cam(
                        input_tensor=img_tensor
                    )[0]

                    visualization = (
                        show_cam_on_image(
                            rgb_img,
                            grayscale_cam,
                            use_rgb=True,
                        )
                    )

                    encoded = (
                        self._encode_image_array(
                            visualization
                        )
                    )

                    results.append(
                        {
                            "method":
                                name,

                            "heatmap_base64":
                                encoded,

                            "overlay_base64":
                                encoded,
                        }
                    )

                    try:
                        cam.clear_hooks()
                    except Exception:
                        pass

                except Exception as cam_error:

                    logger.warning(
                        f"{name} failed: "
                        f"{cam_error}"
                    )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            if not results:

                logger.warning(
                    "All GradCAM methods failed. "
                    "Using fallback visualization."
                )

                return self._mock_gradcam(
                    image
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

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

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
        ) * 100.0

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

        # ====================================================
        # DICE
        # ====================================================

        # IMPORTANT:
        #
        # Dice requires:
        #
        #   predicted_mask
        #   +
        #   ground_truth_mask
        #
        # A newly uploaded MRI does not contain its
        # ground-truth mask.
        #
        # Therefore Dice MUST be None.
        #
        # Never calculate Dice using the predicted mask
        # against itself, because that would always produce
        # 1.0 and would be completely misleading.

        dice_score = None

        # ====================================================
        # RESULT
        # ====================================================

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
                dice_score,

            "dice_available":
                False,

            "dice_note":
                "Dice score is unavailable because "
                "no ground-truth segmentation mask was "
                "provided with this uploaded MRI.",

            "tumor_area_pixels":
                tumor_pixels,

            "tumor_area_percentage":
                float(
                    tumor_percentage
                ),

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

        Used only when trained models are unavailable.
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
                + rng.randn(
                    len(settings.TUMOR_CLASSES)
                ) * 0.05
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
            all_probs.mean(
                axis=0
            )
        )

        mean_probs = np.clip(
            mean_probs,
            1e-10,
            1.0,
        )

        mean_probs /= mean_probs.sum()

        top_probability = float(
            mean_probs.max()
        )

        pred_entropy = float(
            -np.sum(
                mean_probs
                * np.log2(
                    mean_probs
                )
            )
        )

        expected_entropy = float(
            np.mean(
                [
                    -np.sum(
                        p
                        * np.log2(
                            np.clip(
                                p,
                                1e-10,
                                1.0,
                            )
                        )
                    )
                    for p in all_probs
                ]
            )
        )

        mutual_info = max(
            0.0,
            float(
                pred_entropy
                - expected_entropy
            ),
        )

        confidence_threshold = float(
            getattr(
                settings,
                "UNCERTAINTY_THRESHOLD",
                0.75,
            )
        )

        is_uncertain = bool(
            top_probability
            < confidence_threshold
            or mutual_info > 0.30
        )

        uncertainty = {

            "method":
                "monte_carlo_dropout",

            "num_samples":
                settings.MC_DROPOUT_SAMPLES,

            "predictive_entropy":
                pred_entropy,

            "mutual_information":
                mutual_info,

            "confidence":
                top_probability,

            "top_probability":
                top_probability,

            "is_uncertain":
                is_uncertain,

            "uncertainty_reason":
                (
                    "low_class_probability"
                    if top_probability
                    < confidence_threshold
                    else
                    "high_epistemic_uncertainty"
                    if mutual_info > 0.30
                    else
                    "none"
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

        size = settings.SEGMENTATION_SIZE

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

                "dice_available":
                    False,

                "dice_note":
                    "Dice score is unavailable "
                    "without a ground-truth mask.",

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

            "dice_available":
                False,

            "dice_note":
                "Dice score is unavailable "
                "without a ground-truth mask.",

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
