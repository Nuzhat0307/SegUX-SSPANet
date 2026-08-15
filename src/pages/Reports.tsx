import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { FileText, Download, Loader2, Search, Filter } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { PredictionResult, Patient, TUMOR_CLASS_DISPLAY } from '../lib/types'
import { formatDate } from '../lib/utils'
import { generateReportPDF } from '../lib/pdfReport'
import PageHeader from '../components/PageHeader'
import { PredictionBadge } from '../components/PredictionBadge'

interface ListPrediction {
  id: string; patient_id: string; predicted_class: PredictionResult['predicted_class']; predicted_class_display: string
  uncertainty: PredictionResult['uncertainty']; model_version: string; inference_time_ms: number; created_at: string
}
interface ReportRow { id: string; prediction_id: string; report_type: string; created_at: string }

export default function Reports() {
  const [predictions, setPredictions] = useState<ListPrediction[]>([])
  const [reports, setReports] = useState<ReportRow[]>([])
  const [patients, setPatients] = useState<Map<string, Patient>>(new Map())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generatingId, setGeneratingId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState('all')

  const loadData = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [{ data: predData, error: predError }, { data: repData, error: repError }] = await Promise.all([
        supabase.from('predictions').select('id,patient_id,predicted_class,predicted_class_display,uncertainty,model_version,inference_time_ms,created_at').order('created_at', { ascending: false }),
        supabase.from('reports').select('id,prediction_id,report_type,created_at').order('created_at', { ascending: false }),
      ])
      if (predError) throw predError
      if (repError) throw repError
      const preds = (predData || []) as ListPrediction[]
      const reps = (repData || []) as ReportRow[]
      const ids = [...new Set(preds.map(p => p.patient_id).filter(Boolean))]
      const { data: patData, error: patError } = ids.length ? await supabase.from('patients').select('id,name,age,gender,mrn,notes,created_at').in('id', ids) : { data: [], error: null }
      if (patError) throw patError
      const map = new Map<string, Patient>()
      ;(patData || []).forEach((p: any) => map.set(p.id, p as Patient))
      setPredictions(preds); setReports(reps); setPatients(map)
    } catch (e: any) {
      console.error('Reports load error:', e); setError(e?.message || 'Unable to load reports')
      setPredictions([]); setReports([]); setPatients(new Map())
    } finally { setLoading(false) }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleGenerateReport = async (predictionId: string) => {
    setGeneratingId(predictionId); setError(null)
    try {
      const { data, error: fetchError } = await supabase.from('predictions').select('*').eq('id', predictionId).single()
      if (fetchError) throw fetchError
      const prediction = data as unknown as PredictionResult
      const patient = patients.get(prediction.patient_id)
      if (!patient) throw new Error('Patient record is unavailable for this scan.')
      await generateReportPDF(prediction, patient)
      const { data: existing, error: existingError } = await supabase.from('reports').select('id, prediction_id, report_type, created_at').eq('prediction_id', predictionId).eq('report_type', 'full').order('created_at', { ascending: false }).limit(1).maybeSingle()
      if (existingError) throw existingError
      if (!existing) {
        const { data: newReport, error: insertError } = await supabase.from('reports').insert({ prediction_id: predictionId, report_type: 'full' }).select('id,prediction_id,report_type,created_at').single()
        if (insertError) throw insertError
        setReports(prev => [newReport as ReportRow, ...prev])
      }
    } catch (e: any) {
      console.error('Report generation failed:', e); setError(e?.message || 'Report generation failed')
    } finally { setGeneratingId(null) }
  }

  const filteredPredictions = predictions.filter(p => {
    if (classFilter !== 'all' && p.predicted_class !== classFilter) return false
    if (search.trim()) {
      const patient = patients.get(p.patient_id)
      const text = `${patient?.name || ''} ${patient?.mrn || ''} ${p.predicted_class_display}`.toLowerCase()
      if (!text.includes(search.trim().toLowerCase())) return false
    }
    return true
  })
  const reportedIds = new Set(reports.map(r => r.prediction_id))
  const classOptions = [{ value:'all', label:'All Types' }, { value:'glioma', label:'Glioma' }, { value:'meningioma', label:'Meningioma' }, { value:'pituitary', label:'Pituitary' }, { value:'no_tumor', label:'No Tumor' }]

  return <div>
    <PageHeader title="Reports" description="Generate and download professional PDF diagnostic reports" />
    <div className="mb-6 card p-4"><div className="flex flex-wrap items-center gap-3">
      <div className="relative flex-1 min-w-[200px]"><Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" /><input value={search} onChange={e=>setSearch(e.target.value)} className="input-field pl-10" placeholder="Search by patient name, MRN, or diagnosis..." /></div>
      <div className="flex items-center gap-2"><Filter className="h-4 w-4 text-neutral-400" /><select value={classFilter} onChange={e=>setClassFilter(e.target.value)} className="input-field w-auto">{classOptions.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</select></div>
    </div></div>

    {error && <div className="mb-6 rounded-lg bg-error-50 px-4 py-3 text-sm text-error-700">{error}</div>}

    {reports.length > 0 && <div className="mb-8"><h2 className="mb-4 text-lg font-semibold text-neutral-900">Generated Reports</h2><div className="card divide-y divide-neutral-200">
      {reports.slice(0,10).map(rep => { const p=predictions.find(x=>x.id===rep.prediction_id); const patient=p?patients.get(p.patient_id):undefined; return <div key={rep.id} className="flex items-center justify-between p-4"><div className="flex items-center gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success-100"><FileText className="h-5 w-5 text-success-600" /></div><div><p className="text-sm font-semibold text-neutral-900">{patient?.name || 'Patient record unavailable'}</p><p className="text-xs text-neutral-500">{patient?.mrn || '—'} — {p ? TUMOR_CLASS_DISPLAY[p.predicted_class] : '—'} — {formatDate(rep.created_at)}</p></div></div>{p && <div className="flex gap-2"><Link to={`/prediction/${p.id}`} className="btn-secondary text-xs">View</Link><button onClick={()=>handleGenerateReport(p.id)} disabled={generatingId===p.id} className="btn-secondary text-xs">{generatingId===p.id?<Loader2 className="h-3.5 w-3.5 animate-spin"/>:<Download className="h-3.5 w-3.5"/>}Download</button></div>}</div> })}
    </div></div>}

    <div><h2 className="mb-4 text-lg font-semibold text-neutral-900">Available Scans</h2>{loading ? <div className="space-y-3">{[1,2,3].map(i=><div key={i} className="h-20 rounded-lg shimmer-bg"/>)}</div> : filteredPredictions.length ? <div className="space-y-3">{filteredPredictions.map(pred=>{const patient=patients.get(pred.patient_id); const hasReport=reportedIds.has(pred.id); return <div key={pred.id} className="card flex items-center justify-between p-4"><div className="min-w-0"><div className="flex items-center gap-2"><p className="truncate text-sm font-semibold text-neutral-900">{patient?.name || 'Patient record unavailable'}</p>{hasReport&&<span className="badge bg-success-100 text-success-700"><FileText className="h-3 w-3"/> Reported</span>}</div><p className="text-xs text-neutral-500">{patient?.mrn || '—'} — {formatDate(pred.created_at)}</p><div className="mt-1.5"><PredictionBadge tumorClass={pred.predicted_class} confidence={pred.uncertainty?.confidence??0} isUncertain={pred.uncertainty?.is_uncertain??false} size="sm"/></div></div><div className="flex shrink-0 items-center gap-3"><Link to={`/prediction/${pred.id}`} className="text-sm font-medium text-primary-600 hover:text-primary-700">View</Link><button onClick={()=>handleGenerateReport(pred.id)} disabled={generatingId===pred.id} className="btn-primary text-xs">{generatingId===pred.id?<><Loader2 className="h-3.5 w-3.5 animate-spin"/>Generating...</>:<><Download className="h-3.5 w-3.5"/>{hasReport?'Regenerate':'Generate Report'}</>}</button></div></div>})}</div> : <div className="flex flex-col items-center justify-center py-20 text-center"><div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-neutral-100"><FileText className="h-8 w-8 text-neutral-400"/></div><h2 className="text-lg font-semibold text-neutral-700">No scans available</h2><p className="mt-1 text-sm text-neutral-400">{predictions.length ? 'No scans match the selected filters' : 'Upload an MRI scan to generate diagnostic reports'}</p><Link to="/upload" className="btn-primary mt-4">Upload MRI</Link></div>}</div>
  </div>
}
