"""Production inference service for the trained SegUX-SSPANet checkpoint.

Unlike the legacy development inference service, this module never falls back
to synthetic/mock predictions. Classification follows the same staged pipeline
used during training: U-Net segmentation guidance is generated first and then
passed into SegUX-SSPANet for classification and MC-Dropout uncertainty.
"""

import io
import time
import base64
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from loguru import logger

from app.core.config import settings
from app.services.inference import InferenceService


class _GuidedClassifier:
    """Small torch wrapper used by GradCAM to keep inference guidance fixed."""

    def __init__(self, classifier, guidance):
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self, model, fixed_guidance):
                super().__init__()
                self.model = model
                self.fixed_guidance = fixed_guidance

            def forward(self, x):
                return self.model(x, self.fixed_guidance)

        self.module = Wrapper(classifier, guidance)


class TrainedInferenceService(InferenceService):
    """Inference service that requires the real trained checkpoint."""

    def load_models(self):
        import os
        import torch
        from ml.models.segux_sspanet import SegUXSSPANet
        from ml.models.unet import UNet

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = settings.MODEL_CHECKPOINT_PATH

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Trained checkpoint not found: {checkpoint_path}. "
                "Place segux_sspanet_best.pth at this path before starting the API."
            )

        try:
            # The checkpoint contains the complete classifier state, so there is
            # no need to download ImageNet weights during inference.
            self._model = SegUXSSPANet(
                num_classes=settings.NUM_CLASSES,
                backbone="resnet50",
                pretrained=False,
            ).to(self._device)
            self._segmentor = UNet(in_channels=1, out_channels=1).to(self._device)

            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            classifier_state = checkpoint.get("classifier", checkpoint)
            self._model.load_state_dict(classifier_state, strict=True)

            segmentor_state = checkpoint.get("segmentor")
            if segmentor_state is None:
                raise RuntimeError(
                    "The trained checkpoint does not contain the U-Net 'segmentor' state. "
                    "Segmentation-guided inference cannot run safely."
                )
            self._segmentor.load_state_dict(segmentor_state, strict=True)

            self._model.eval()
            self._segmentor.eval()
            self._loaded = True
            logger.info(
                f"TRAINED MODEL LOADED SUCCESSFULLY: {checkpoint_path} on {self._device}"
            )
        except Exception:
            self._model = None
            self._segmentor = None
            self._loaded = False
            logger.exception(f"Failed to load trained checkpoint: {checkpoint_path}")
            raise

    async def predict(self, image_base64: str, patient_id: str) -> Dict[str, Any]:
        if not self._loaded or self._model is None or self._segmentor is None:
            raise RuntimeError(
                "Trained SegUX-SSPANet checkpoint is not loaded. "
                "Prediction has been blocked; no mock inference is permitted."
            )

        start_time = time.time()
        image_data = self._decode_base64_image(image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert("L")
        result = self._real_trained_inference(image_pil)
        result["inference_time_ms"] = int((time.time() - start_time) * 1000)
        result["model_version"] = settings.MODEL_VERSION
        result["patient_id"] = patient_id
        result["inference_source"] = "trained_checkpoint"
        result["checkpoint_path"] = settings.MODEL_CHECKPOINT_PATH
        return result

    def _real_trained_inference(self, image: Image.Image) -> Dict[str, Any]:
        import torch

        img_array = np.array(
            image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            dtype=np.float32,
        )
        img_tensor = (
            torch.from_numpy(img_array)
            .float()
            .unsqueeze(0)
            .repeat(3, 1, 1)
            .unsqueeze(0)
            .to(self._device)
            / 255.0
        )

        # Match training: U-Net generates guidance before classification.
        seg_input = (
            torch.from_numpy(
                np.array(
                    image.resize(
                        (settings.SEGMENTATION_SIZE, settings.SEGMENTATION_SIZE)
                    ),
                    dtype=np.float32,
                )
            )
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
            / 255.0
        )

        with torch.no_grad():
            seg_mask_tensor = torch.sigmoid(self._segmentor(seg_input))
            guidance = torch.nn.functional.interpolate(
                seg_mask_tensor,
                size=img_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

            logits = self._model(img_tensor, guidance)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            seg_mask = seg_mask_tensor.cpu().numpy()[0, 0]

        predicted_idx = int(np.argmax(probs))
        predicted_class = settings.TUMOR_CLASSES[predicted_idx]
        probabilities = [
            {
                "label": cls,
                "display_name": self._display_name(cls),
                "probability": float(probs[i]),
            }
            for i, cls in enumerate(settings.TUMOR_CLASSES)
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        uncertainty = self._guided_mc_dropout_uncertainty(img_tensor, guidance)
        segmentation = self._build_segmentation_result(seg_mask, image)
        gradcam_results = self._guided_gradcam(img_tensor, guidance, image)
        feature_explanation = self._generate_feature_explanation(
            predicted_class,
            probabilities,
            image,
            segmentation,
        )

        return {
            "predicted_class": predicted_class,
            "predicted_class_display": self._display_name(predicted_class),
            "probabilities": probabilities,
            "uncertainty": uncertainty,
            "segmentation": segmentation,
            "gradcam_results": gradcam_results,
            "feature_explanation": feature_explanation,
        }

    @staticmethod
    def _display_name(label: str) -> str:
        return {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "pituitary": "Pituitary Tumor",
            "no_tumor": "No Tumor",
        }[label]

    def _guided_mc_dropout_uncertainty(self, img_tensor, guidance):
        import torch

        self._model.eval()
        for module in self._model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(settings.MC_DROPOUT_SAMPLES):
                logits = self._model(img_tensor, guidance)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])

        self._model.eval()
        all_probs = np.asarray(all_probs)
        mean_probs = all_probs.mean(axis=0)
        pred_entropy = -np.sum(mean_probs * np.log2(mean_probs + 1e-10))
        expected_entropy = np.mean(
            [-np.sum(p * np.log2(p + 1e-10)) for p in all_probs]
        )
        mutual_info = max(0.0, pred_entropy - expected_entropy)
        confidence = 1.0 - pred_entropy / np.log2(settings.NUM_CLASSES)

        return {
            "method": "monte_carlo_dropout",
            "num_samples": settings.MC_DROPOUT_SAMPLES,
            "predictive_entropy": float(pred_entropy),
            "mutual_information": float(mutual_info),
            "confidence": float(confidence),
            "is_uncertain": bool(
                confidence < settings.UNCERTAINTY_THRESHOLD or mutual_info > 0.3
            ),
        }

    def _guided_gradcam(self, img_tensor, guidance, image: Image.Image) -> List[Dict[str, str]]:
        try:
            import numpy as np
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            wrapper = _GuidedClassifier(self._model, guidance).module
            target_layer = self._model.get_target_layer()
            methods = [
                ("gradcam", GradCAM),
                ("gradcam_plus_plus", GradCAMPlusPlus),
                ("eigengradcam", EigenGradCAM),
            ]
            rgb_img = np.float32(
                np.array(
                    image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)).convert("RGB")
                )
                / 255.0
            )
            results = []
            for name, cam_class in methods:
                cam = cam_class(model=wrapper, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=img_tensor)[0]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                encoded = self._encode_image_array(visualization)
                results.append(
                    {
                        "method": name,
                        "heatmap_base64": encoded,
                        "overlay_base64": encoded,
                    }
                )
            return results
        except Exception as exc:
            logger.warning(f"GradCAM computation unavailable: {exc}")
            return []

    def _build_segmentation_result(self, mask: np.ndarray, image: Image.Image) -> Dict[str, Any]:
        """Build segmentation result without fabricating a Dice score."""
        result = super()._build_segmentation_result(mask, image)
        result["dice_score"] = None
        return result


inference_service = TrainedInferenceService()
