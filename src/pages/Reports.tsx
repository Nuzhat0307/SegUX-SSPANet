import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  FileText,
  Download,
  Loader2,
  AlertCircle,
  Search,
  Filter,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import { PredictionResult, Patient, TUMOR_CLASS_DISPLAY } from '../lib/types'
import { formatDate } from '../lib/utils'
import { generateReportPDF } from '../lib/pdfReport'
import PageHeader from '../components/PageHeader'
import { PredictionBadge } from '../components/PredictionBadge'

interface ReportRow {
  id: string
  prediction_id: string
  report_type: string
  created_at: string
  prediction?: PredictionResult
  patient?: Patient
}

export default function Reports() {
  const [reports, setReports] = useState<ReportRow[]>([])
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [patients, setPatients] = useState<Map<string, Patient>>(new Map())
  const [loading, setLoading] = useState(true)
  const [generatingId, setGeneratingId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState('all')

const loadData = useCallback(async () => {
  setLoading(true)

  try {
    // Load all predictions
    const { data: predData, error: predError } = await supabase
      .from('predictions')
      .select('*')
      .order('created_at', { ascending: false })

    if (predError) {
      throw predError
    }

    const preds = (predData || []) as unknown as PredictionResult[]
    setPredictions(preds)

    // ---------------------------------------------------------
    // Load patients and create the map BEFORE attaching patients
    // to reports.
    // ---------------------------------------------------------
    const patMap = new Map<string, Patient>()

    if (preds.length > 0) {
      const patientIds = [
        ...new Set(
          preds
            .map((p) => p.patient_id)
            .filter(Boolean),
        ),
      ]

      if (patientIds.length > 0) {
        const { data: patData, error: patError } = await supabase
          .from('patients')
          .select('*')
          .in('id', patientIds)

        if (patError) {
          throw patError
        }

        ;(patData || []).forEach((p: any) => {
          patMap.set(p.id, p as Patient)
        })
      }
    }

    // Update React state after building the map
    setPatients(patMap)

    // ---------------------------------------------------------
    // Load existing reports
    // ---------------------------------------------------------
    const { data: repData, error: repError } = await supabase
      .from('reports')
      .select('*')
      .order('created_at', { ascending: false })

    if (repError) {
      throw repError
    }

    const reps = (repData || []) as unknown as ReportRow[]

    // ---------------------------------------------------------
    // Attach prediction + patient to every report
    // IMPORTANT: use patMap, NOT patients state
    // ---------------------------------------------------------
    for (const rep of reps) {
      rep.prediction = preds.find(
        (p) => p.id === rep.prediction_id,
      )

      if (rep.prediction) {
        rep.patient = patMap.get(
          rep.prediction.patient_id,
        )
      }
    }

    setReports(reps)
  } catch (err) {
    console.error('Reports load error:', err)
  } finally {
    setLoading(false)
  }
}, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const handleGenerateReport = async (prediction: PredictionResult) => {
    const patient = patients.get(prediction.patient_id)
    if (!patient) return
    setGeneratingId(prediction.id)
    try {
      await generateReportPDF(prediction, patient)
      // Record it
      const { data: newReport } = await supabase
        .from('reports')
        .insert({
          prediction_id: prediction.id,
          report_type: 'full',
        })
        .select()
        .single()
      if (newReport) {
        const rep = newReport as unknown as ReportRow
        rep.prediction = prediction
        rep.patient = patient
        setReports((prev) => [rep, ...prev])
      }
    } catch (err) {
      console.error('Report generation failed:', err)
    } finally {
      setGeneratingId(null)
    }
  }

  const filteredPredictions = predictions.filter((p) => {
    if (classFilter !== 'all' && p.predicted_class !== classFilter) return false
    if (search) {
      const patient = patients.get(p.patient_id)
      const searchText = `${patient?.name || ''} ${patient?.mrn || ''} ${p.predicted_class_display}`.toLowerCase()
      if (!searchText.includes(search.toLowerCase())) return false
    }
    return true
  })

  const reportedPredictionIds = new Set(reports.map((r) => r.prediction_id))

  const classOptions = [
    { value: 'all', label: 'All Types' },
    { value: 'glioma', label: 'Glioma' },
    { value: 'meningioma', label: 'Meningioma' },
    { value: 'pituitary', label: 'Pituitary' },
    { value: 'no_tumor', label: 'No Tumor' },
  ]

  return (
    <div>
      <PageHeader
        title="Reports"
        description="Generate and download professional PDF diagnostic reports"
      />

      {/* Filters */}
      <div className="mb-6 card p-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input-field pl-10"
              placeholder="Search by patient name, MRN, or diagnosis..."
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-neutral-400" />
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="input-field w-auto"
            >
              {classOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Already generated reports */}
      {reports.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-4 text-lg font-semibold text-neutral-900">Generated Reports</h2>
          <div className="card divide-y divide-neutral-200">
            {reports.slice(0, 10).map((rep) => (
              <div key={rep.id} className="flex items-center justify-between p-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success-100">
                    <FileText className="h-5 w-5 text-success-600" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-neutral-900">
                      {rep.patient?.name || 'Unknown patient'}
                    </p>
                    <p className="text-xs text-neutral-500">
                      {rep.patient?.mrn} — {TUMOR_CLASS_DISPLAY[rep.prediction?.predicted_class || 'no_tumor']} — {formatDate(rep.created_at)}
                    </p>
                  </div>
                </div>
                {rep.prediction && rep.patient && (
                  <button
                    onClick={() => {
                      handleGenerateReport(rep.prediction!)
                    }}
                    disabled={generatingId === rep.prediction_id}
                    className="btn-secondary text-xs"
                  >
                    {generatingId === rep.prediction_id ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Download className="h-3.5 w-3.5" />
                    )}
                    Download
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Available predictions for report generation */}
      <div>
        <h2 className="mb-4 text-lg font-semibold text-neutral-900">Available Scans</h2>
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 rounded-lg shimmer-bg" />
            ))}
          </div>
        ) : filteredPredictions.length > 0 ? (
          <div className="space-y-3">
            {filteredPredictions.map((pred) => {
              const patient = patients.get(pred.patient_id)
              const hasReport = reportedPredictionIds.has(pred.id)
              return (
                <div key={pred.id} className="card flex items-center justify-between p-4">
                  <div className="flex items-center gap-4">
                    <img
                      src={pred.image_base64}
                      alt="MRI"
                      className="h-16 w-16 rounded-lg object-cover"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-semibold text-neutral-900">
                          {patient?.name || 'Unknown'}
                        </p>
                        {hasReport && (
                          <span className="badge bg-success-100 text-success-700">
                            <FileText className="h-3 w-3" /> Reported
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-neutral-500">
                        {patient?.mrn} — {formatDate(pred.created_at)}
                      </p>
                      <div className="mt-1.5">
                        <PredictionBadge
                          tumorClass={pred.predicted_class}
                          confidence={pred.uncertainty.confidence}
                          isUncertain={pred.uncertainty.is_uncertain}
                          size="sm"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Link
                      to={`/prediction/${pred.id}`}
                      className="text-sm font-medium text-primary-600 hover:text-primary-700"
                    >
                      View
                    </Link>
                    <button
                      onClick={() => handleGenerateReport(pred)}
                      disabled={generatingId === pred.id}
                      className="btn-primary text-xs"
                    >
                      {generatingId === pred.id ? (
                        <>
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Download className="h-3.5 w-3.5" />
                          {hasReport ? 'Regenerate' : 'Generate Report'}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-neutral-100">
              <FileText className="h-8 w-8 text-neutral-400" />
            </div>
            <h2 className="text-lg font-semibold text-neutral-700">No scans available</h2>
            <p className="mt-1 text-sm text-neutral-400">
              Upload an MRI scan to generate diagnostic reports
            </p>
            <Link to="/upload" className="btn-primary mt-4">
              Upload MRI
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
