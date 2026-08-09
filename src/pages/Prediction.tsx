import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import {
  ArrowLeft,
  Download,
  Brain,
  Scan,
  Layers,
  Flame,
  Gauge,
  Clock,
  AlertTriangle,
  CheckCircle,
  FileText,
  Loader2,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import {
  PredictionResult,
  Patient,
  TumorClass,
  TUMOR_CLASS_DISPLAY,
  TUMOR_CLASS_COLORS,
  TUMOR_CLASS_DESCRIPTIONS,
  GradCAMResult,
} from '../lib/types'
import { formatProbability, formatConfidence, formatDate, getUncertaintyLabel } from '../lib/utils'
import { PredictionBadge, ConfidenceBadge } from '../components/PredictionBadge'
import { generateReportPDF } from '../lib/pdfReport'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const GRADCAM_LABELS: Record<string, string> = {
  gradcam: 'GradCAM',
  gradcam_plus_plus: 'GradCAM++',
  eigengradcam: 'EigenGradCAM',
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
    async function loadPrediction() {
      if (!id) return
      setLoading(true)
      try {
        const { data, error } = await supabase
          .from('predictions')
          .select('*')
          .eq('id', id)
          .maybeSingle()

        if (error) throw error
        if (!data) {
          setLoading(false)
          return
        }

        const pred = data as unknown as PredictionResult
        setPrediction(pred)

        // Load patient
        const { data: pat } = await supabase
          .from('patients')
          .select('*')
          .eq('id', pred.patient_id)
          .maybeSingle()
        setPatient(pat as unknown as Patient)
      } catch (err) {
        console.error('Error loading prediction:', err)
      } finally {
        setLoading(false)
      }
    }
    loadPrediction()
  }, [id])

  const handleDownloadReport = async () => {
    if (!prediction || !patient) return
    setGeneratingPDF(true)
    try {
      await generateReportPDF(prediction, patient)
      // Record report generation
      await supabase.from('reports').insert({
        prediction_id: prediction.id,
        report_type: 'full',
      })
    } catch (err) {
      console.error('Report generation failed:', err)
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
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <AlertTriangle className="mb-4 h-12 w-12 text-neutral-300" />
        <h2 className="text-lg font-semibold text-neutral-700">Prediction not found</h2>
        <Link to="/history" className="btn-primary mt-4">
          View History
        </Link>
      </div>
    )
  }

  const tumorColor = TUMOR_CLASS_COLORS[prediction.predicted_class]
  const uncertaintyInfo = getUncertaintyLabel(prediction.uncertainty.confidence)
  const chartData = prediction.probabilities.map((p) => ({
    name: p.displayName,
    value: parseFloat((p.probability * 100).toFixed(1)),
    color: TUMOR_CLASS_COLORS[p.label],
  }))

  const currentGradCAM: GradCAMResult | undefined = prediction.gradcam_results[activeGradCAM]

  return (
    <div>
      {/* Header */}
      <div className="mb-6 flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(-1)}
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-neutral-200 bg-white text-neutral-600 transition-all hover:bg-neutral-50"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-neutral-900">Analysis Results</h1>
            <p className="mt-0.5 text-sm text-neutral-500">
              {patient?.name} (MRN: {patient?.mrn}) — {formatDate(prediction.created_at)}
            </p>
          </div>
        </div>
        <button
          onClick={handleDownloadReport}
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

      {/* Primary result banner */}
      <div
        className="mb-6 rounded-xl border-l-4 p-6"
        style={{
          borderLeftColor: tumorColor,
          backgroundColor: `${tumorColor}08`,
        }}
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <PredictionBadge
                tumorClass={prediction.predicted_class}
                confidence={prediction.uncertainty.confidence}
                isUncertain={prediction.uncertainty.is_uncertain}
              />
              <ConfidenceBadge
                confidence={prediction.uncertainty.confidence}
                isUncertain={prediction.uncertainty.is_uncertain}
              />
            </div>
            <h2 className="text-xl font-bold text-neutral-900">
              {TUMOR_CLASS_DISPLAY[prediction.predicted_class]}
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-neutral-600">
              {TUMOR_CLASS_DESCRIPTIONS[prediction.predicted_class]}
            </p>
          </div>
          <div className="flex items-center gap-4 text-sm text-neutral-500">
            <div className="flex items-center gap-1.5">
              <Clock className="h-4 w-4" />
              {prediction.inference_time_ms}ms
            </div>
            <div className="flex items-center gap-1.5">
              <Brain className="h-4 w-4" />
              {prediction.model_version}
            </div>
          </div>
        </div>
      </div>

      {/* Uncertainty warning */}
      {prediction.uncertainty.is_uncertain && (
        <div className="mb-6 flex items-center gap-3 rounded-xl border border-warning-200 bg-warning-50 px-5 py-4">
          <AlertTriangle className="h-5 w-5 flex-shrink-0 text-warning-600" />
          <div>
            <p className="text-sm font-semibold text-warning-800">
              Uncertainty Detected — Expert Review Recommended
            </p>
            <p className="mt-0.5 text-sm text-warning-700">
  {prediction.uncertainty.uncertainty_reason === 'high_epistemic_uncertainty'
    ? `The model's confidence is ${formatConfidence(
        prediction.uncertainty.confidence
      )}, with elevated epistemic uncertainty (mutual information ${
        prediction.uncertainty.mutual_information
      .toFixed(4)}). This case may benefit from review by a specialist.`
    : `The model's confidence is ${formatConfidence(
        prediction.uncertainty.confidence
      )}, which is below the confidence threshold. This case may benefit from review by a specialist.`}
</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Original MRI + Segmentation */}
        <div className="card overflow-hidden">
          <div className="flex items-center gap-2 border-b border-neutral-200 px-5 py-3">
            <Scan className="h-4 w-4 text-primary-600" />
            <h3 className="text-sm font-semibold text-neutral-900">MRI Scan & Segmentation</h3>
          </div>
          <div className="grid grid-cols-2 gap-px bg-neutral-200">
            <div className="bg-neutral-900 p-4">
              <p className="mb-2 text-center text-xs font-medium text-neutral-400">Original MRI</p>
              <img
                src={prediction.image_base64}
                alt="Original MRI"
                className="mx-auto max-h-64 rounded-lg object-contain"
              />
            </div>
            <div className="bg-neutral-900 p-4">
              <p className="mb-2 text-center text-xs font-medium text-neutral-400">
                Segmentation Overlay (U-Net)
              </p>
              <img
                src={prediction.segmentation.overlay_base64}
                alt="Segmentation overlay"
                className="mx-auto max-h-64 rounded-lg object-contain"
              />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-px border-t border-neutral-200 bg-neutral-200">
            <div className="bg-white px-4 py-3 text-center">
              <p className="text-xs text-neutral-500">Dice Score</p>
              <p className="text-lg font-bold text-primary-600">
  {prediction.segmentation.dice_score != null
    ? prediction.segmentation.dice_score.toFixed(3)
    : 'N/A'}
</p>
            </div>
            <div className="bg-white px-4 py-3 text-center">
              <p className="text-xs text-neutral-500">Tumor Area</p>
              <p className="text-lg font-bold text-neutral-900">
                {prediction.segmentation.tumor_area_percentage.toFixed(1)}%
              </p>
            </div>
            <div className="bg-white px-4 py-3 text-center">
              <p className="text-xs text-neutral-500">Bounding Box</p>
              <p className="text-sm font-semibold text-neutral-700">
                {prediction.segmentation.bounding_box
                  ? `${prediction.segmentation.bounding_box.width}×${prediction.segmentation.bounding_box.height}`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        {/* Classification probabilities */}
        <div className="card p-5">
          <div className="mb-4 flex items-center gap-2">
            <Gauge className="h-4 w-4 text-primary-600" />
            <h3 className="text-sm font-semibold text-neutral-900">
              Classification Probabilities
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={90} />
              <Tooltip
                formatter={(value: any) => `${value}%`}
                contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-2">
            {prediction.probabilities.map((p) => (
              <div key={p.label} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: TUMOR_CLASS_COLORS[p.label] }}
                  />
                  <span className="font-medium text-neutral-700">{p.displayName}</span>
                </div>
                <span className="font-mono font-semibold text-neutral-900">
                  {formatProbability(p.probability)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* GradCAM Results */}
        <div className="card overflow-hidden lg:col-span-2">
          <div className="flex items-center justify-between border-b border-neutral-200 px-5 py-3">
            <div className="flex items-center gap-2">
              <Flame className="h-4 w-4 text-accent-600" />
              <h3 className="text-sm font-semibold text-neutral-900">
                Explainability — GradCAM Visualizations
              </h3>
            </div>
            <div className="flex gap-1 rounded-lg bg-neutral-100 p-1">
              {prediction.gradcam_results.map((g, idx) => (
                <button
                  key={g.method}
                  onClick={() => setActiveGradCAM(idx)}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
                    activeGradCAM === idx
                      ? 'bg-white text-primary-700 shadow-sm'
                      : 'text-neutral-500 hover:text-neutral-700'
                  }`}
                >
                  {GRADCAM_LABELS[g.method] || g.method}
                </button>
              ))}
            </div>
          </div>
          {currentGradCAM && (
            <div className="grid grid-cols-1 gap-px bg-neutral-200 sm:grid-cols-2">
              <div className="bg-neutral-900 p-4">
                <p className="mb-2 text-center text-xs font-medium text-neutral-400">
                  Heatmap ({GRADCAM_LABELS[currentGradCAM.method]})
                </p>
                <img
                  src={currentGradCAM.heatmap_base64}
                  alt={`${currentGradCAM.method} heatmap`}
                  className="mx-auto max-h-72 rounded-lg object-contain"
                />
              </div>
              <div className="bg-neutral-900 p-4">
                <p className="mb-2 text-center text-xs font-medium text-neutral-400">
                  Overlay on Original MRI
                </p>
                <img
                  src={currentGradCAM.overlay_base64}
                  alt={`${currentGradCAM.method} overlay`}
                  className="mx-auto max-h-72 rounded-lg object-contain"
                />
              </div>
            </div>
          )}
          <div className="border-t border-neutral-200 px-5 py-3">
            <p className="text-xs text-neutral-500">
              GradCAM visualizations highlight the image regions most influential to the model's
              prediction. GradCAM++ provides finer-grained object localization, while EigenGradCAM
              uses principal component analysis of gradients for more stable explanations.
            </p>
          </div>
        </div>

        {/* Uncertainty Estimation */}
        <div className="card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center gap-2">
            <Layers className="h-4 w-4 text-secondary-600" />
            <h3 className="text-sm font-semibold text-neutral-900">
              Uncertainty Estimation — Monte Carlo Dropout
            </h3>
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-lg bg-neutral-50 p-4 text-center">
              <p className="text-xs text-neutral-500">Confidence</p>
              <p
                className={`mt-1 text-xl font-bold ${
                  uncertaintyInfo.color === 'text-success-700'
                    ? 'text-success-600'
                    : uncertaintyInfo.color === 'text-warning-700'
                    ? 'text-warning-600'
                    : 'text-error-600'
                }`}
              >
                {formatConfidence(prediction.uncertainty.confidence)}
              </p>
            </div>
            <div className="rounded-lg bg-neutral-50 p-4 text-center">
              <p className="text-xs text-neutral-500">Predictive Entropy</p>
              <p className="mt-1 text-xl font-bold text-neutral-900">
                {prediction.uncertainty.predictive_entropy.toFixed(4)}
              </p>
            </div>
            <div className="rounded-lg bg-neutral-50 p-4 text-center">
              <p className="text-xs text-neutral-500">Mutual Information</p>
              <p className="mt-1 text-xl font-bold text-neutral-900">
                {prediction.uncertainty.mutual_information.toFixed(4)}
              </p>
            </div>
            <div className="rounded-lg bg-neutral-50 p-4 text-center">
              <p className="text-xs text-neutral-500">MC Samples</p>
              <p className="mt-1 text-xl font-bold text-neutral-900">
                {prediction.uncertainty.num_samples}
              </p>
            </div>
          </div>
          <div className="mt-4 rounded-lg bg-primary-50 px-4 py-3">
            <p className="text-xs text-primary-800">
              <strong>Method:</strong> Monte Carlo Dropout runs {prediction.uncertainty.num_samples}{' '}
              stochastic forward passes through the network with dropout enabled at inference time.
              Predictive entropy measures total uncertainty; mutual information captures epistemic
              (model) uncertainty. High mutual information suggests the model needs more training
              data for this type of input.
            </p>
          </div>
        </div>
      </div>

      {/* Action bar */}
      <div className="mt-6 flex flex-wrap justify-between gap-3">
        <Link to="/upload" className="btn-secondary">
          <Scan className="h-4 w-4" />
          New Scan
        </Link>
        <div className="flex gap-3">
          <Link to="/history" className="btn-secondary">
            <FileText className="h-4 w-4" />
            View History
          </Link>
          <button onClick={handleDownloadReport} disabled={generatingPDF} className="btn-primary">
            <Download className="h-4 w-4" />
            Download PDF Report
          </button>
        </div>
      </div>
    </div>
  )
}
