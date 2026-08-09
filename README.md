# SegUX-SSPANet Brain Tumor Diagnosis System

An AI-powered, hospital-ready web application for brain tumor diagnosis from MRI scans. The system integrates **segmentation-guided attention learning** with **uncertainty-aware deep learning** and **explainable AI** to provide clinicians with transparent, trustworthy diagnostic support.

## Overview

SegUX-SSPANet extends the SSPANet (Strip-Style Pooling Attention Network) framework with:

- **Multi-task learning**: Simultaneous tumor classification + segmentation
- **Segmentation-guided attention**: U-Net masks guide the attention maps for better localization
- **Uncertainty estimation**: Monte Carlo Dropout quantifies prediction confidence
- **Multi-method explainability**: GradCAM, GradCAM++, and EigenGradCAM visualizations
- **Clinical decision support**: Flags low-confidence cases for expert review
- **Downloadable PDF reports**: Professional diagnostic reports with all findings

### Tumor Classes

| Class | Description |
|-------|-------------|
| Glioma | Tumor originating in glial cells |
| Meningioma | Tumor on brain/spinal cord membranes |
| Pituitary | Tumor of the pituitary gland |
| No Tumor | Normal scan — no tumor detected |

---

## Architecture

```
SegUX-SSPANet/
├── src/                          # Frontend (React + Vite + Tailwind)
│   ├── components/               # Shared UI components
│   ├── lib/                      # Supabase client, auth, types, mock inference, PDF
│   ├── pages/                    # Dashboard, Upload, Prediction, History, Reports, Settings, Login, Register
│   └── App.tsx                   # Router and protected routes
│
├── backend/                      # Backend (FastAPI + PyTorch)
│   ├── app/
│   │   ├── api/routes.py         # REST API endpoints
│   │   ├── core/                 # Config, security (JWT), database
│   │   ├── models/               # SQLAlchemy ORM models
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── inference.py      # Full inference pipeline
│   │   │   └── report_generator.py
│   │   └── main.py               # FastAPI app factory
│   ├── ml/
│   │   ├── models/
│   │   │   ├── sspanet.py        # Strip-Style Pooling Attention Network
│   │   │   ├── segux_sspanet.py  # Full multi-task model (ResNet50 + SSPANet)
│   │   │   └── unet.py           # U-Net for segmentation
│   │   ├── utils/
│   │   │   ├── losses.py         # Dice Loss, BCE-Dice, Multi-task loss
│   │   │   ├── metrics.py        # Classification, segmentation, uncertainty metrics
│   │   │   └── gradcam_utils.py  # GradCAM/GradCAM++/EigenGradCAM
│   │   └── data/dataset.py       # Figshare + BraTS dataset loaders
│   ├── scripts/
│   │   ├── train.py              # Training pipeline
│   │   ├── evaluate.py           # Evaluation script
│   │   └── download_dataset.py   # Dataset download + preprocessing
│   └── requirements.txt
│
├── docker/                       # Nginx config
├── Dockerfile                    # Frontend Docker
├── docker-compose.yml            # Full deployment
└── README.md
```

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+ (for backend)
- Docker and Docker Compose (for containerized deployment)

### 1. Frontend (Web Application)

```bash
# Install dependencies
npm install

# Start the dev server
npm run dev
```

The app will be available at `http://localhost:5173`.

**Note**: The frontend works standalone with Supabase for auth and data storage. The in-browser mock inference engine produces realistic AI results (classification, segmentation, GradCAM, uncertainty) without requiring the Python backend — perfect for demos and development.

### 2. Backend (FastAPI + PyTorch)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your Supabase credentials

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with Swagger docs at `http://localhost:8000/api/v1/docs`.

### 3. Docker Deployment (Full Stack)

```bash
# Build and run all services
docker-compose up --build

# Or with local PostgreSQL
docker-compose --profile local-db up --build
```

- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/api/v1/docs`

---

## Training the AI Model

### Step 1: Download Datasets

```bash
cd backend
source venv/bin/activate

# Download and preprocess Figshare dataset
python -m scripts.download_dataset --dataset figshare --output ml/data

# Download and preprocess BraTS (requires manual download)
# See instructions in the script output
python -m scripts.download_dataset --dataset brats --output ml/data
```

**Figshare Brain Tumor Dataset**:
- Auto-downloads from figshare.com
- ~3,064 MRI images across 3 tumor classes
- URL: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

**BraTS (Brain Tumor Segmentation)**:
- Requires free registration at https://www.synapse.org/brats
- Provides multimodal MRI volumes with pixel-level segmentation masks
- The script extracts 2D slices from NIfTI volumes

### Step 2: Train

```bash
# Full training (classification + segmentation)
python -m scripts.train \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --figshare_dir ml/data/figshare \
    --brats_dir ml/data/brats_2d \
    --checkpoint_dir ml/checkpoints

# Classification only (if BraTS not available)
python -m scripts.train \
    --epochs 50 \
    --figshare_dir ml/data/figshare
```

The training pipeline:
1. Loads ResNet50 with ImageNet-pretrained weights
2. Attaches SSPANet attention modules at each residual stage
3. Trains classification head with CrossEntropy loss
4. Trains U-Net segmentor with BCE + Dice loss
5. Uses CosineAnnealing learning rate scheduling
6. Saves the best checkpoint based on validation accuracy

### Step 3: Evaluate

```bash
python -m scripts.evaluate \
    --checkpoint ml/checkpoints/segux_sspanet_best.pth \
    --figshare_dir ml/data/figshare \
    --brats_dir ml/data/brats_2d \
    --output ml/eval_results.json
```

Evaluation computes:
- **Classification**: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
- **Segmentation**: Dice Score, IoU, Sensitivity, Specificity
- **Uncertainty**: Brier Score, Expected Calibration Error (ECE)
- **MC Dropout**: Average confidence, predictive entropy, uncertain case ratio

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register a new user |
| POST | `/api/v1/auth/login` | Login and get JWT token |
| GET | `/api/v1/auth/me` | Get current user info |

### Patients

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/patients` | Create a patient |
| GET | `/api/v1/patients` | List all patients |
| GET | `/api/v1/patients/{id}` | Get a specific patient |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Run AI inference on an MRI |
| GET | `/api/v1/predictions` | List all predictions |
| GET | `/api/v1/predictions/{id}` | Get a specific prediction |

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/reports/{prediction_id}` | Generate a PDF report |
| GET | `/api/v1/reports` | List all reports |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check + model status |

---

## AI Model Components

### SSPANet (Strip-Style Pooling Attention Network)

The core attention module from the research paper, combining three pooling streams:

1. **Channel Attention** (Squeeze-Excitation): Captures inter-channel dependencies
2. **Strip Pooling**: Horizontal and vertical 1D pooling for long-range spatial dependencies
3. **Style Pooling**: Mean + variance pooling for texture and style features

The three streams are fused via a 1x1 convolution and added as a residual connection.

**File**: `backend/ml/models/sspanet.py` — **Adapted from the paper's architecture.**

### SegUX-SSPANet (Full Model)

Multi-task architecture:
- **Backbone**: ResNet50 (ImageNet-pretrained)
- **Attention**: SSPANet modules at each residual stage (256, 512, 1024, 2048 channels)
- **Classification head**: Global avg pool → FC → Dropout → FC(num_classes)
- **MC Dropout**: Dropout enabled at inference for uncertainty estimation

**File**: `backend/ml/models/segux_sspanet.py` — **Newly written.**

### U-Net (Segmentation)

Standard U-Net encoder-decoder for binary tumor segmentation:
- Encoder: 4 down-sampling blocks (64 → 128 → 256 → 512 → 1024 channels)
- Decoder: 4 up-sampling blocks with skip connections
- Output: 1-channel sigmoid probability map

**File**: `backend/ml/models/unet.py` — **Adapted from the standard U-Net architecture (Ronneberger et al., 2015).**

### GradCAM Variants

Three explainability methods:
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **GradCAM++**: Weighted gradients for better localization
- **EigenGradCAM**: PCA-based gradients for stability

**File**: `backend/ml/utils/gradcam_utils.py` — **Uses the `pytorch-grad-cam` library.**

### Monte Carlo Dropout Uncertainty

- Runs N stochastic forward passes (default: 30) with dropout enabled
- **Predictive entropy**: Total uncertainty (H[p(y|x)])
- **Mutual information**: Epistemic uncertainty (model uncertainty)
- **Confidence**: 1 - normalized entropy
- Cases with confidence < 0.75 or high mutual information are flagged for expert review

### Loss Functions

- **CrossEntropy**: Classification loss
- **Dice Loss**: Soft Dice for segmentation
- **BCE + Dice**: Combined segmentation loss for stable training
- **Multi-task loss**: Weighted sum of classification + segmentation losses

**File**: `backend/ml/utils/losses.py` — **Newly written.**

---

## Database Schema

The system uses Supabase (PostgreSQL) with the following tables:

| Table | Description |
|-------|-------------|
| `patients` | Patient demographics (name, age, gender, MRN) |
| `predictions` | AI inference results (class, probabilities, uncertainty, segmentation, GradCAM) |
| `reports` | PDF report generation metadata |

All tables have **Row Level Security (RLS)** enabled — users can only access their own data.

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Login | `/login` | Email/password authentication |
| Register | `/register` | New account creation |
| Dashboard | `/dashboard` | Overview stats, recent analyses, tumor distribution chart |
| Upload MRI | `/upload` | Drag-and-drop MRI upload with patient info form |
| Prediction | `/prediction/:id` | Full results: classification, segmentation, GradCAM, uncertainty |
| History | `/history` | Searchable, filterable list of all analyses |
| Reports | `/reports` | Generate and download PDF diagnostic reports |
| Settings | `/settings` | Profile management and model configuration |

---

## Component Attribution

| Component | Source | Notes |
|-----------|--------|-------|
| ResNet50 | torchvision (pretrained) | ImageNet weights |
| U-Net | Standard architecture | Adapted from Ronneberger et al. 2015 |
| GradCAM/++/EigenGradCAM | pytorch-grad-cam library | pip install grad-cam |
| SSPANet | Research paper | Custom implementation of the paper's architecture |
| SegUX-SSPANet | Newly written | Multi-task extension combining all components |
| MC Dropout | Standard technique | Adapted for medical uncertainty estimation |
| Dice Loss | Standard | Soft Dice for differentiable segmentation metrics |
| FastAPI backend | Newly written | Full REST API with JWT auth |
| React frontend | Newly written | Complete medical UI with Tailwind CSS |

---

## Disclaimer

This system is intended for **research and educational purposes only**. It is NOT a substitute for professional medical diagnosis. All AI-generated findings must be validated by a qualified radiologist or neurologist before any clinical decision is made.

---

## License

This project is provided for research and educational use.
