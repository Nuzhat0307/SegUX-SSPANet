export type TumorClass = 'glioma' | 'meningioma' | 'pituitary' | 'no_tumor'

export interface ClassPrediction {
  label: TumorClass
  displayName: string
  probability: number
}

export interface UncertaintyEstimate {
  method: 'monte_carlo_dropout'
  num_samples: number
  predictive_entropy: number
  mutual_information: number
  confidence: number
  is_uncertain: boolean
}

export interface GradCAMResult {
  method: 'gradcam' | 'gradcam_plus_plus' | 'eigengradcam'
  heatmap_base64: string
  overlay_base64: string
}

export interface SegmentationResult {
  mask_base64: string
  overlay_base64: string
  dice_score: number
  tumor_area_pixels: number
  tumor_area_percentage: number
  bounding_box: { x: number; y: number; width: number; height: number } | null
}

export interface PredictionResult {
  id: string
  patient_id: string
  image_url: string
  image_base64: string
  predicted_class: TumorClass
  predicted_class_display: string
  probabilities: ClassPrediction[]
  uncertainty: UncertaintyEstimate
  segmentation: SegmentationResult
  gradcam_results: GradCAMResult[]
  model_version: string
  inference_time_ms: number
  created_at: string
  notes?: string
}

export interface Patient {
  id: string
  name: string
  age: number | null
  gender: 'male' | 'female' | 'other' | null
  mrn: string
  created_at: string
  prediction_count?: number
}

export interface UserSession {
  id: string
  email: string
  fullName?: string
}

export const TUMOR_CLASS_DISPLAY: Record<TumorClass, string> = {
  glioma: 'Glioma',
  meningioma: 'Meningioma',
  pituitary: 'Pituitary Tumor',
  no_tumor: 'No Tumor',
}

export const TUMOR_CLASS_COLORS: Record<TumorClass, string> = {
  glioma: '#dc2626',
  meningioma: '#f97316',
  pituitary: '#8b5cf6',
  no_tumor: '#22c55e',
}

export const TUMOR_CLASS_DESCRIPTIONS: Record<TumorClass, string> = {
  glioma:
    'A tumor originating in the glial cells of the brain or spine. Gliomas can range from low-grade to high-grade and require prompt clinical evaluation.',
  meningioma:
    'A typically slow-growing tumor that forms on the membranes covering the brain and spinal cord. Most meningiomas are benign but may require monitoring or surgical removal.',
  pituitary:
    'A tumor of the pituitary gland at the base of the brain. These can affect hormone levels and may require endocrinological assessment.',
  no_tumor:
    'No tumor detected in the provided MRI scan. The model classified this image as normal with high confidence.',
}
