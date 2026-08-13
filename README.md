# SegUX-SSPANet — Brain Tumor Diagnosis System

**SegUX-SSPANet** is an AI-assisted brain tumor analysis system that combines MRI classification, tumor segmentation, segmentation-guided feature learning, uncertainty estimation, and explainable AI into a web-based diagnostic support application.

The project integrates a **ResNet-50 backbone**, **SSPANet attention modules**, a **U-Net segmentation model**, **Monte Carlo Dropout**, and **Grad-CAM-based explainability** with a React/FastAPI application.

> **Research and educational project:** This system is not intended to replace diagnosis by a qualified medical professional.

---

## 1. Project Overview

SegUX-SSPANet is designed to analyze brain MRI images and provide:

* Brain tumor type classification
* Tumor-region segmentation
* Segmentation-guided classification
* Prediction confidence and uncertainty estimation
* Grad-CAM, Grad-CAM++, and EigenGradCAM visualizations
* Patient and prediction history
* PDF diagnostic reports
* Authentication and protected application routes
* REST APIs for inference and report generation

The system currently supports four model output classes:

| Class ID | Class      |
| -------: | ---------- |
|        0 | Glioma     |
|        1 | Meningioma |
|        2 | Pituitary  |
|        3 | No Tumor   |

The class mapping is defined directly in the dataset implementation and application configuration.

### Important Dataset Note

Although the model configuration defines four classes, the local Figshare dataset used during development contained:

* Glioma: 1,426 images
* Meningioma: 708 images
* Pituitary: 930 images
* No Tumor: 0 images

Therefore, the practical training/evaluation data used for the current development version did not contain `no_tumor` samples.

---

## 2. Key Features

### MRI Classification

The SegUX-SSPANet classifier uses a ResNet-50 backbone with SSPANet attention modules to classify MRI images into the configured tumor categories.

### Tumor Segmentation

A U-Net model is trained separately using BraTS-derived 2D MRI slices and corresponding segmentation masks.

The current processed BraTS dataset used during development contains:

* **4,798 processed image slices**
* **4,798 corresponding segmentation masks**

The training script intentionally uses a smaller subset for laptop-friendly training.

### Segmentation-Guided Classification

The segmentation model is not trained jointly with the classifier.

Instead:

```text
BraTS images + masks
        │
        ▼
      U-Net
        │
        ▼
Best segmentation model
        │
        ▼
Frozen during classification
        │
        ▼
Segmentation guidance
        │
        ▼
Figshare MRI ───────────────┐
                            ▼
                     SegUX-SSPANet
                            │
                            ▼
                     Tumor prediction
```

For Figshare classification images, the trained U-Net:

1. Converts the 3-channel image to grayscale.
2. Resizes it from `224 × 224` to `256 × 256`.
3. Generates a segmentation probability map.
4. Resizes the segmentation guidance back to `224 × 224`.
5. Passes the original image and segmentation guidance to SegUX-SSPANet.

This behavior is implemented directly in `train.py` and `evaluate.py`.

---

# 3. System Architecture

```text
                         ┌───────────────────────┐
                         │      React / Vite      │
                         │   Frontend Interface   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      FastAPI API       │
                         │ Authentication / REST  │
                         └───────────┬───────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
             Prediction          Patients           Reports
                  │
                  ▼
        ┌───────────────────────┐
        │   SegUX-SSPANet        │
        │ ResNet50 + SSPANet     │
        └──────────┬────────────┘
                   │
                   │ Segmentation guidance
                   ▼
             ┌───────────┐
             │   U-Net   │
             │ Segmentor │
             └───────────┘
                   │
                   ▼
            Tumor Region Mask

Additional AI services:
        │
        ├── MC Dropout
        ├── Grad-CAM
        ├── Grad-CAM++
        └── EigenGradCAM

Data / Storage:
        │
        └── Supabase / PostgreSQL
```

---

# 4. Repository Structure

```text
SegUX-SSPANet/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   └── report_generator.py
│   │   └── main.py
│   │
│   ├── ml/
│   │   ├── models/
│   │   │   ├── sspanet.py
│   │   │   ├── segux_sspanet.py
│   │   │   └── unet.py
│   │   │
│   │   ├── data/
│   │   │   └── dataset.py
│   │   │
│   │   └── utils/
│   │       ├── losses.py
│   │       ├── metrics.py
│   │       └── gradcam_utils.py
│   │
│   ├── scripts/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── download_dataset.py
│   │
│   └── requirements.txt
│
├── src/
│   ├── components/
│   ├── lib/
│   ├── pages/
│   └── App.tsx
│
├── public/
├── supabase/
│   └── migrations/
│
├── docker/
├── Dockerfile
├── docker-compose.yml
├── package.json
├── vite.config.ts
└── README.md
```

The repository currently contains separate frontend, backend, ML, Supabase, and Docker components.

---

# 5. AI Model Architecture

## 5.1 ResNet-50 Backbone

The classification network uses an ImageNet-pretrained ResNet-50 backbone.

The feature stages are augmented with SSPANet attention modules.

The implementation uses the feature dimensions:

```text
256 → 512 → 1024 → 2048
```

The final classification head performs global average pooling followed by fully connected layers and dropout.

---

## 5.2 SSPANet Attention

SSPANet is implemented using three complementary feature-processing streams:

### Channel Attention

Captures relationships between feature channels.

### Strip Pooling

Uses horizontal and vertical pooling to capture longer-range spatial information.

### Style Pooling

Uses statistical feature information such as mean and variance to represent texture/style characteristics.

The resulting features are fused and incorporated into the network through a residual connection.

Implementation:

```text
backend/ml/models/sspanet.py
```

---

# 6. U-Net Segmentation

The segmentation component is a standard U-Net encoder-decoder architecture.

```text
Input MRI
   │
   ▼
Encoder
64 → 128 → 256 → 512 → 1024
   │
   ▼
Bottleneck
   │
   ▼
Decoder
1024 → 512 → 256 → 128 → 64
   │
   ▼
1-channel output
   │
   ▼
Tumor probability mask
```

The U-Net receives grayscale images at `256 × 256` resolution and produces a one-channel tumor probability map.

Implementation:

```text
backend/ml/models/unet.py
```

---

# 7. Training Methodology

The current implementation uses **staged training**, rather than end-to-end simultaneous optimization of both networks.

## Stage 1 — U-Net Segmentation Training

BraTS-derived image-mask pairs are used to train the U-Net.

Current configured training subset:

```text
Training samples:   500
Validation samples: 200
Image size:         256 × 256
Batch size:         16
Learning rate:      1 × 10⁻⁴
Optimizer:          AdamW
Weight decay:       1 × 10⁻⁴
Epochs:             20
```

The segmentation objective uses BCE + Dice loss.

The best U-Net checkpoint is selected according to validation Dice score:

```text
ml/checkpoints/segux_unet_best.pth
```

The implementation also creates recovery checkpoints during segmentation training.

---

## Stage 2 — SegUX-SSPANet Classification Training

Figshare images are used for classification.

Current configured subset:

```text
Training samples:   1,000
Validation samples: 200
Image size:         224 × 224
Batch size:         16
Learning rate:      1 × 10⁻⁴
Optimizer:          AdamW
Weight decay:       1 × 10⁻⁴
Epochs:             50
```

During this stage, the U-Net parameters are frozen.

For every Figshare image:

```text
MRI image
    │
    ├──────────────► SegUX-SSPANet
    │
    ▼
 Frozen U-Net
    │
    ▼
Segmentation guidance
    │
    └──────────────► SegUX-SSPANet
                         │
                         ▼
                  Tumor classification
```

The classifier is optimized using CrossEntropyLoss.

The best complete model is selected according to validation classification accuracy.

---

# 8. Training Configuration

| Parameter                    | Current Configuration            |
| ---------------------------- | -------------------------------- |
| Classification backbone      | ResNet-50                        |
| Attention                    | SSPANet                          |
| Segmentation model           | U-Net                            |
| Classification image size    | 224 × 224                        |
| Segmentation image size      | 256 × 256                        |
| Figshare training samples    | 1,000                            |
| Figshare validation samples  | 200                              |
| BraTS training samples       | 500                              |
| BraTS validation samples     | 200                              |
| Classification epochs        | 50                               |
| Segmentation epochs          | 20                               |
| Batch size                   | 16                               |
| Classification learning rate | 1e-4                             |
| Segmentation learning rate   | 1e-4                             |
| Optimizer                    | AdamW                            |
| Weight decay                 | 1e-4                             |
| LR scheduler                 | CosineAnnealingLR                |
| Classification loss          | CrossEntropyLoss                 |
| Segmentation loss            | BCE + Dice                       |
| Default device               | CUDA if available, otherwise CPU |

These values correspond to the current implementation rather than theoretical/projected settings.
The training pipeline is configured for 20 U-Net segmentation epochs and 50 SegUX-SSPANet classification epochs. During development, earlier training runs were stopped after fewer epochs, including a 3-epoch run. Therefore, the configured epoch counts should not be interpreted as the number of epochs completed in every experiment.

---

# 9. Dataset Preparation

## Figshare Brain Tumor Dataset

The Figshare dataset is used for **classification**.

The dataset loader searches class directories for:

```text
.jpg
.jpeg
.png
```

Expected structure:

```text
ml/data/figshare/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```

The loader deterministically shuffles the available samples and limits the training/validation subsets to the configured sample counts for laptop-friendly execution.

---

## BraTS Dataset

BraTS is used for **segmentation**.

The project processes MRI volumes into 2D image-mask pairs.

Expected processed structure:

```text
ml/data/brats_2d/
├── images/
└── masks/
```

The current development dataset contains:

```text
Images: 4,798
Masks:  4,798
```

Only a subset of these processed samples is loaded for training by default.

---

# 10. Evaluation

The project provides a dedicated evaluation script:

```bash
python -m scripts.evaluate \
    --checkpoint ml/checkpoints/segux_sspanet_best.pth \
    --figshare_dir ml/data/figshare \
    --brats_dir ml/data/brats_2d \
    --output ml/eval_results.json
```

The evaluation script uses a smaller test subset to make CPU/laptop evaluation practical.

Default configuration:

```text
Maximum test samples per dataset: 300
Batch size:                       2
MC Dropout samples:               30
```

The test pool is formed from samples remaining after the training and validation subsets.

---

# 11. Evaluation Metrics

## Classification

The evaluation pipeline calculates:

* Accuracy
* Precision
* Recall
* Macro F1-score
* AUC-ROC
* Confusion matrix

## Segmentation

The evaluation pipeline calculates:

* Dice score
* IoU
* Sensitivity
* Specificity

## Uncertainty

The system evaluates:

* Brier score
* Expected Calibration Error (ECE)
* MC-Dropout confidence
* Predictive entropy
* Mutual information
* Uncertain-case ratio

These metrics are calculated by the actual evaluation code and saved to the generated evaluation JSON.

---

# 12. Uncertainty Estimation

SegUX-SSPANet uses **Monte Carlo Dropout** to estimate predictive uncertainty.

The default configuration performs:

```text
30 stochastic forward passes
```

For each prediction, the system calculates:

* Mean predicted probabilities
* Maximum predicted-class probability
* Predictive entropy
* Expected entropy
* Mutual information

A case can be considered uncertain when:

```text
confidence < 0.75
```

or when:

```text
mutual information > 0.30
```

This information is intended to support expert review rather than replace clinical judgment.

---

# 13. Explainable AI

The system provides three Grad-CAM-based visualization methods:

### Grad-CAM

Highlights image regions that contribute to the selected prediction.

### Grad-CAM++

Provides improved localization using weighted gradients.

### EigenGradCAM

Uses principal-component-based gradient information to produce activation maps.

Implementation:

```text
backend/ml/utils/gradcam_utils.py
```

The project uses the `pytorch-grad-cam` library for these explainability methods.

---

# 14. Web Application

The frontend is implemented using:

* React
* TypeScript
* Vite
* Tailwind CSS

The application includes the following pages:

| Page       | Purpose                            |
| ---------- | ---------------------------------- |
| Login      | User authentication                |
| Register   | Account creation                   |
| Dashboard  | Overview and recent analyses       |
| Upload MRI | MRI upload and patient information |
| Prediction | AI analysis results                |
| History    | Previous predictions               |
| Reports    | PDF report generation/download     |
| Settings   | Profile/application settings       |

The current repository contains these frontend routes and components.

---

# 15. Backend

The backend uses:

* Python
* FastAPI
* PyTorch
* Pydantic
* SQLAlchemy
* PostgreSQL/Supabase
* JWT-based authentication

The primary inference endpoint is:

```text
POST /api/v1/predict
```

The backend performs model loading, preprocessing, segmentation guidance generation, classification, uncertainty estimation, explainability processing, and prediction persistence.

---

# 16. API Endpoints

## Authentication

```text
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
```

## Patients

```text
POST /api/v1/patients
GET  /api/v1/patients
GET  /api/v1/patients/{id}
```

## Predictions

```text
POST /api/v1/predict
GET  /api/v1/predictions
GET  /api/v1/predictions/{id}
```

## Reports

```text
POST /api/v1/reports/{prediction_id}
GET  /api/v1/reports
```

## Health

```text
GET /api/v1/health
```

The API structure is defined in the current backend implementation.

---

# 17. Database

The application uses PostgreSQL through Supabase.

Main application entities include:

```text
patients
predictions
reports
```

Patient records contain information such as patient name and demographic information.

Prediction records store AI-generated analysis information, while reports store PDF report metadata.

---

# 18. Local Setup

## Prerequisites

Recommended environment:

```text
Python 3.11
Node.js 18+
npm
PostgreSQL / Supabase
Git
```

Python 3.11 is recommended for compatibility with the project's machine-learning dependencies.

---

## Frontend

From the project root:

```powershell
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

---

## Backend

```powershell
cd backend
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/api/v1/docs
```

---

# 19. Environment Variables

Create the required `.env` configuration in the backend.

Typical configuration includes:

```env
DATABASE_URL=your_postgresql_connection_string

SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

SECRET_KEY=your_secure_secret_key
```

Do not commit real Supabase keys, service-role keys, passwords, or secret keys to GitHub.

---

# 20. Training

After preparing the datasets:

```powershell
cd backend
venv\Scripts\activate
```

Run the training pipeline:

```powershell
python -m scripts.train `
    --epochs 50 `
    --seg_epochs 20 `
    --batch_size 16 `
    --lr 1e-4 `
    --seg_lr 1e-4 `
    --figshare_dir ml/data/figshare `
    --brats_dir ml/data/brats_2d `
    --checkpoint_dir ml/checkpoints
```

The training pipeline performs:

```text
Stage 1
BraTS
  ↓
U-Net training
  ↓
Best U-Net checkpoint
  ↓
Freeze U-Net

Stage 2
Figshare
  ↓
Generate U-Net guidance
  ↓
SegUX-SSPANet classification
  ↓
Best complete checkpoint
```

The final checkpoint is:

```text
backend/ml/checkpoints/segux_sspanet_best.pth
```

The checkpoint stores the classifier, segmentor, validation metrics, epoch, and model version.

---

# 21. Evaluation

Run:

```powershell
python -m scripts.evaluate `
    --checkpoint ml/checkpoints/segux_sspanet_best.pth `
    --figshare_dir ml/data/figshare `
    --brats_dir ml/data/brats_2d `
    --output ml/eval_results.json
```

For CPU-based evaluation:

```powershell
python -m scripts.evaluate `
    --checkpoint ml/checkpoints/segux_sspanet_best.pth `
    --figshare_dir ml/data/figshare `
    --brats_dir ml/data/brats_2d `
    --output ml/eval_results.json `
    --device cpu
```

The resulting file contains the measured evaluation metrics.

---

# 22. Model Checkpoints

Important generated files include:

```text
ml/checkpoints/
├── segux_unet_best.pth
├── segmentation_recovery.pth
└── segux_sspanet_best.pth
```

`segux_sspanet_best.pth` is the primary complete model checkpoint containing both classifier and U-Net weights.

---

# 23. Docker

The repository contains Docker configuration for deployment:

```text
Dockerfile
docker-compose.yml
docker/
```

The intended full-stack architecture is:

```text
Browser
   │
   ▼
Frontend container
   │
   ▼
FastAPI backend
   │
   ├── PyTorch models
   └── Database
```

Docker support is included in the repository, although local Python/Node execution is useful for development and debugging.

---

# 24. Current Project Limitations

The current implementation has several important research limitations.

### Limited Training Subsets

The code deliberately restricts training to:

```text
Figshare: 1,000 training + 200 validation
BraTS:    500 training + 200 validation
```

This was chosen to make training feasible on a laptop.

### Segmentation Training Size

Only 500 BraTS samples are currently used by the training script, even though the processed dataset contains substantially more image-mask pairs.

Increasing the number of training samples may improve segmentation generalization, provided the additional samples are representative and properly processed.

### Class Imbalance / Missing No-Tumor Samples

The model defines four classes, but the current local Figshare dataset contained no `no_tumor` images.

Consequently, four-class support exists in the model architecture, but the current trained dataset does not provide representative training examples for that class.

### Staged Rather Than End-to-End Training

The current implementation does not jointly optimize the U-Net and classification network in a single end-to-end multitask optimization loop.

The U-Net is trained first and subsequently frozen while generating segmentation guidance for classifier training.

### Evaluation Subset

The default evaluation uses a maximum of 300 test samples from each dataset to reduce CPU execution time.

Therefore, evaluation results should be interpreted as results on the configured test subset, not as a full-dataset clinical benchmark.

---

# 25. Reproducibility

The project records the main model configuration in the application settings:

```text
Model version: SegUX-SSPANet-v1.0.0
Image size: 224
Segmentation size: 256
Number of classes: 4
MC Dropout samples: 30
Uncertainty threshold: 0.75
Classification epochs: 50
Learning rate: 1e-4
Batch size: 16
```

The current configuration is defined in:

```text
backend/app/core/config.py
```

and the staged training behavior is implemented in:

```text
backend/scripts/train.py
```

---

# 26. Research Results

Model performance should be reported using the metrics generated by the evaluation pipeline rather than manually entered or estimated values.

Run:

```powershell
python -m scripts.evaluate `
    --checkpoint ml/checkpoints/segux_sspanet_best.pth `
    --figshare_dir ml/data/figshare `
    --brats_dir ml/data/brats_2d `
    --output ml/eval_results.json
```

The generated results include:

```text
Classification
├── Accuracy
├── Precision
├── Recall
├── Macro F1
├── AUC-ROC
└── Confusion Matrix

Segmentation
├── Dice
├── IoU
├── Sensitivity
└── Specificity

Uncertainty
├── Brier Score
├── ECE
├── MC Confidence
├── Predictive Entropy
├── Mutual Information
└── Uncertain Case Ratio
```

No performance value is hard-coded in this README because the reported values should correspond to the actual checkpoint and evaluation run.

---

# 27. Technology Stack

| Layer           | Technology                           |
| --------------- | ------------------------------------ |
| Frontend        | React + TypeScript                   |
| Build Tool      | Vite                                 |
| Styling         | Tailwind CSS                         |
| Backend         | FastAPI                              |
| Deep Learning   | PyTorch                              |
| Classification  | ResNet-50 + SSPANet                  |
| Segmentation    | U-Net                                |
| Explainability  | Grad-CAM / Grad-CAM++ / EigenGradCAM |
| Uncertainty     | Monte Carlo Dropout                  |
| Database        | PostgreSQL / Supabase                |
| Authentication  | JWT / Supabase                       |
| PDF Reports     | Backend report generation            |
| Deployment      | Docker / Docker Compose              |
| Version Control | Git / GitHub                         |

---

# 28. Project Workflow

```text
User Login
    │
    ▼
Patient Registration
    │
    ▼
MRI Upload
    │
    ▼
Preprocessing
    │
    ▼
U-Net Segmentation
    │
    ├──────────────► Tumor Mask
    │
    ▼
Segmentation Guidance
    │
    ▼
SegUX-SSPANet
    │
    ├──────────────► Tumor Classification
    │
    ├──────────────► Classification Probabilities
    │
    ├──────────────► GradCAM Visualizations
    │
    └──────────────► MC-Dropout Uncertainty
                            │
                            ▼
                    Clinical Review Flag
                            │
                            ▼
                     Prediction Record
                            │
                            ▼
                      PDF Report
```

---

# 29. Future Improvements

Potential future improvements include:

* Training U-Net on a larger BraTS subset or full processed dataset
* Adding more balanced `no_tumor` samples
* Performing end-to-end joint optimization
* Increasing the classification training dataset
* GPU-based training and evaluation
* Patient-level rather than image-level dataset splitting
* More rigorous external validation
* Calibration improvement
* Ablation studies for SSPANet and segmentation guidance
* Comparison with baseline CNN architectures
* Cross-dataset generalization experiments
* Clinical expert validation

---

# 30. Important Scientific Note

This project demonstrates an AI-assisted research workflow for brain MRI analysis.

It should **not** be described as a clinically validated diagnostic system unless appropriate clinical validation, external testing, regulatory review, and expert assessment have been completed.

The outputs are intended to assist research and development and should not be used as the sole basis for medical decisions.

---

# 31. License

This project is provided for research and educational purposes.

---

## Author

**Nuzhat Sultana**

GitHub: [Nuzhat0307](https://github.com/Nuzhat0307)

Project: **SegUX-SSPANet**
