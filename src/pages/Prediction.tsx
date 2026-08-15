import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import {
  ArrowLeft,
  Download,
  Layers,
  AlertTriangle,
  Loader2,
  Lightbulb,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import {
  PredictionResult,
  Patient,
  TUMOR_CLASS_DISPLAY,
  TUMOR_CLASS_COLORS,
  TUMOR_CLASS_DESCRIPTIONS,
  GradCAMResult,
  FeatureExplanation,
} from '../lib/types'
import { normalizePredictionResult } from '../lib/mockInference'
import { formatProbability, formatConfidence, formatDate } from '../lib/utils'
import { PredictionBadge, ConfidenceBadge } from '../components/PredictionBadge'
import { generateReportPDF } from '../lib/pdfReport'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const GRADCAM_LABELS: Record<string, string> = {
  gradcam: 'GradCAM',
  gradcam_plus_plus: 'GradCAM++',
  eigengradcam: 'EigenGradCAM',
}

const DISPLAY_FALLBACKS: Record<string, string> = {
  glioma: 'Glioma',
  meningioma: 'Meningioma',
  pituitary: 'Pituitary Tumor',
  no_tumor: 'No Tumor',
}

export default function Prediction() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeGradCAM, setActiveGradCAM] = useState(0)
  const [generatingPDF, setGeneratingPDF] = useState(false)

  useEffect(() => {
    let cancelled = false

    const loadPrediction = async () => {
      if (!id) {
        setLoading(false)
        return
      }

      try {
        const { data, error } = await supabase
          .from('predictions')
          .select('*')
          .eq('id', id)
          .maybeSingle()

        if (error) throw error
        if (!data) {
          if (!cancelled) setPrediction(null)
          return
        }

        const normalized = normalizePredictionResult(data)
        if (cancelled) return

        setPrediction(normalized)

        if (normalized.patient_id) {
          const { data: patientData } = await supabase
            .from('patients')
            .select('*')
            .eq('id', normalized.patient_id)
            .maybeSingle()

          if (!cancelled && patientData) {
            setPatient(patientData as unknown as Patient)
          }
        }
      } catch (error) {
        console.error('Failed to load prediction:', error)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadPrediction()

    return () => {
      cancelled = true
    }
  }, [id])

  const download = async () => {
    if (!prediction || !patient) return

    setGeneratingPDF(true)
    try {
      await generateReportPDF(prediction, patient)

      const { data, error } = await supabase
        .from('reports')
        .select('id')
        .eq('prediction_id', prediction.id)
        .eq('report_type', 'full')
        .limit(1)
        .maybeSingle()

      if (error) throw error

      if (!data) {
        await supabase.from('reports').insert({
          prediction_id: prediction.id,
          report_type: 'full',
        })
      }
    } catch (error) {
      console.error('Report generation failed:', error)
    } finally {
      setGeneratingPDF(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="flex flex-col items-center py-20">
        <AlertTriangle className="mb-4 h-12 w-12 text-neutral-300" />
        <h2 className="text-lg font-semibold">Prediction not found</h2>
        <Link to="/history" className="btn-primary mt-4">
          View History
        </Link>
      </div>
    )
  }

  const color = TUMOR_CLASS_COLORS[prediction.predicted_class]
  const chart = prediction.probabilities.map((probability) => ({
    name:
      probability.displayName ||
      TUMOR_CLASS_DISPLAY[probability.label] ||
      DISPLAY_FALLBACKS[probability.label] ||
      probability.label,
    value: +(probability.probability * 100).toFixed(1),
    color: TUMOR_CLASS_COLORS[probability.label],
  }))

  const grad: GradCAMResult | undefined =
    prediction.gradcam_results?.[activeGradCAM]

  return (
    <div>
      <div className="mb-6 flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(-1)}
            className="flex h-10 w-10 items-center justify-center rounded-lg border bg-white"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold">Analysis Results</h1>
            <p className="text-sm text-neutral-500">
              {patient?.name || 'Unknown Patient'} (MRN: {patient?.mrn || 'N/A'}) —{' '}
              {formatDate(prediction.created_at)}
            </p>
          </div>
        </div>

        <button
          onClick={download}
          disabled={generatingPDF}
          className="btn-primary"
        >
          {generatingPDF ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Download className="h-4 w-4" />
              Download Report
            </>
          )}
        </button>
      </div>

      <div
        className="mb-6 rounded-xl border-l-4 p-6"
        style={{
          borderLeftColor: color,
          backgroundColor: `${color}08`,
        }}
      >
        <PredictionBadge
          tumorClass={prediction.predicted_class}
          confidence={prediction.uncertainty.confidence}
          isUncertain={prediction.uncertainty.is_uncertain}
        />
        <ConfidenceBadge
          confidence={prediction.uncertainty.confidence}
          isUncertain={prediction.uncertainty.is_uncertain}
        />
        <h2 className="mt-2 text-xl font-bold">
          {TUMOR_CLASS_DISPLAY[prediction.predicted_class] ||
            DISPLAY_FALLBACKS[prediction.predicted_class] ||
            prediction.predicted_class}
        </h2>
        <p className="mt-1 text-sm text-neutral-600">
          {TUMOR_CLASS_DESCRIPTIONS[prediction.predicted_class] ||
            'The model prediction should be reviewed together with the MRI and other clinical information.'}
        </p>
      </div>

      {prediction.uncertainty.is_uncertain && (
        <div className="mb-6 rounded-xl border border-warning-200 bg-warning-50 p-4 text-sm text-warning-800">
          <AlertTriangle className="mr-2 inline h-5 w-5" />
          Uncertainty detected — expert review recommended. MC-Dropout confidence:{' '}
          {formatConfidence(prediction.uncertainty.confidence)}.
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card overflow-hidden">
          <div className="grid grid-cols-2 gap-px bg-neutral-200">
            <div className="bg-neutral-900 p-4">
              <p className="mb-2 text-center text-xs text-neutral-400">Original MRI</p>
              <img
                src={prediction.image_base64}
                alt="MRI"
                className="mx-auto max-h-64 object-contain"
              />
            </div>
            <div className="bg-neutral-900 p-4">
              <p className="mb-2 text-center text-xs text-neutral-400">
                Segmentation Overlay
              </p>
              <img
                src={prediction.segmentation.overlay_base64}
                alt="Segmentation"
                className="mx-auto max-h-64 object-contain"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 text-center">
            <div className="p-3">
              <p className="text-xs text-neutral-500">Dice</p>
              <b>{prediction.segmentation.dice_score?.toFixed(3) ?? 'N/A'}</b>
            </div>
            <div className="p-3">
              <p className="text-xs text-neutral-500">Tumor Area</p>
              <b>{prediction.segmentation.tumor_area_percentage.toFixed(1)}%</b>
            </div>
            <div className="p-3">
              <p className="text-xs text-neutral-500">Inference</p>
              <b>{prediction.inference_time_ms}ms</b>
            </div>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="mb-4 font-semibold">Classification Probabilities</h3>

          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chart} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis
                type="category"
                dataKey="name"
                width={120}
                tick={{ fontSize: 12 }}
              />
              <Tooltip
                formatter={(value: number) => [`${value}%`, 'Probability']}
              />
              <Bar dataKey="value">
                {chart.map((entry, index) => (
                  <Cell key={`${entry.name}-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-4 space-y-2">
            {prediction.probabilities.map((probability) => {
              const label =
                probability.displayName ||
                TUMOR_CLASS_DISPLAY[probability.label] ||
                DISPLAY_FALLBACKS[probability.label] ||
                probability.label

              return (
                <div
                  key={probability.label}
                  className="flex justify-between text-sm"
                >
                  <span>{label}</span>
                  <b>{formatProbability(probability.probability)}</b>
                </div>
              )
            })}
          </div>
        </div>

        <div className="card overflow-hidden lg:col-span-2">
          <div className="flex items-center justify-between border-b p-4">
            <h3 className="font-semibold">Explainability — GradCAM</h3>
            <div className="flex gap-1">
              {(prediction.gradcam_results || []).map((result, index) => (
                <button
                  key={result.method}
                  onClick={() => setActiveGradCAM(index)}
                  className={`rounded px-3 py-1 text-xs ${
                    activeGradCAM === index
                      ? 'bg-primary-100 text-primary-700'
                      : 'bg-neutral-100'
                  }`}
                >
                  {GRADCAM_LABELS[result.method] || result.method}
                </button>
              ))}
            </div>
          </div>

          {grad && (
            <div className="grid grid-cols-2 gap-px bg-neutral-200">
              <img
                src={grad.heatmap_base64}
                alt="GradCAM heatmap"
                className="max-h-72 w-full bg-neutral-900 object-contain p-4"
              />
              <img
                src={grad.overlay_base64}
                alt="GradCAM overlay"
                className="max-h-72 w-full bg-neutral-900 object-contain p-4"
              />
            </div>
          )}
        </div>

        {prediction.feature_explanation && (
          <FeatureExplainabilitySection
            explanation={prediction.feature_explanation}
            tumorColor={color}
          />
        )}

        <div className="card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center gap-2">
            <Layers className="h-4 w-4" />
            <h3 className="font-semibold">
              Uncertainty Estimation — Monte Carlo Dropout
            </h3>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div>
              <p className="text-xs text-neutral-500">MC-Dropout Confidence</p>
              <b>{formatConfidence(prediction.uncertainty.confidence)}</b>
            </div>
            <div>
              <p className="text-xs text-neutral-500">Entropy</p>
              <b>{prediction.uncertainty.predictive_entropy.toFixed(4)}</b>
            </div>
            <div>
              <p className="text-xs text-neutral-500">Mutual Information</p>
              <b>{prediction.uncertainty.mutual_information.toFixed(4)}</b>
            </div>
            <div>
              <p className="text-xs text-neutral-500">MC Samples</p>
              <b>{prediction.uncertainty.num_samples}</b>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function FeatureExplainabilitySection({
  explanation,
  tumorColor,
}: {
  explanation: FeatureExplanation
  tumorColor: string
}) {
  return (
    <div className="card p-5 lg:col-span-2">
      <div className="mb-4 flex items-center gap-2">
        <Lightbulb className="h-4 w-4" style={{ color: tumorColor }} />
        <h3 className="font-semibold">Feature-Based Explainability</h3>
      </div>

      <p className="mb-5 text-sm text-neutral-600">{explanation.summary}</p>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase text-neutral-500">
            Detected Features
          </h4>
          {explanation.detected_features?.map((feature) => (
            <div
              key={feature.name}
              className="mb-2 rounded-lg bg-neutral-50 p-3"
            >
              <div className="flex justify-between text-sm">
                <span>{feature.display_name}</span>
                <b>
                  {feature.value}
                  {feature.unit}
                </b>
              </div>
              <p className="mt-1 text-xs text-neutral-500">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase text-neutral-500">
            Key Contributions
          </h4>
          {explanation.key_contributions?.map((contribution) => (
            <div
              key={contribution.feature_name}
              className="mb-2 rounded-lg bg-neutral-50 p-3"
            >
              <div className="flex justify-between text-sm">
                <span>{contribution.display_name}</span>
                <b>{contribution.direction}</b>
              </div>
              <p className="mt-1 text-xs text-neutral-500">
                {contribution.explanation}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-lg border p-3">
          <b className="text-xs text-neutral-500">Region Description</b>
          <p className="mt-1 text-sm">{explanation.region_description}</p>
        </div>
        <div className="rounded-lg border p-3">
          <b className="text-xs text-neutral-500">Clinical Correlation</b>
          <p className="mt-1 text-sm">{explanation.clinical_correlation}</p>
        </div>
      </div>
    </div>
  )
}
