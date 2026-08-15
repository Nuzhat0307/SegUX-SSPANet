import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { FileText, Download, Loader2, Search, Filter } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { PredictionResult, Patient, TUMOR_CLASS_DISPLAY } from '../lib/types'
import { formatDate } from '../lib/utils'
import { generateReportPDF } from '../lib/pdfReport'
import PageHeader from '../components/PageHeader'
import { PredictionBadge } from '../components/PredictionBadge'

interface ListPrediction { id:string; patient_id:string; predicted_class:PredictionResult['predicted_class']; predicted_class_display:string; uncertainty:PredictionResult['uncertainty']; model_version:string; inference_time_ms:number; created_at:string }
interface ReportRow { id:string; prediction_id:string; report_type:string; created_at:string }

export default function Reports() {
  const [predictions,setPredictions]=useState<ListPrediction[]>([])
  const [reports,setReports]=useState<ReportRow[]>([])
  const [patients,setPatients]=useState<Map<string,Patient>>(new Map())
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState<string|null>(null)
  const [generatingId,setGeneratingId]=useState<string|null>(null)
  const [search,setSearch]=useState('')
  const [classFilter,setClassFilter]=useState('all')

  const loadData=useCallback(async()=>{
    setLoading(true); setError(null)
    try {
      // List pages request only metadata. Heavy MRI/segmentation/GradCAM data is
      // fetched only when Generate/Regenerate is clicked.
      const [{data:pd,error:pe},{data:rd,error:re}]=await Promise.all([
        supabase.from('predictions').select('id,patient_id,predicted_class,predicted_class_display,uncertainty,model_version,inference_time_ms,created_at').order('created_at',{ascending:false}),
        supabase.from('reports').select('id,prediction_id,report_type,created_at').order('created_at',{ascending:false}),
      ])
      if(pe) throw pe
      if(re) throw re
      const preds=(pd||[]) as ListPrediction[]
      setPredictions(preds)
      setReports((rd||[]) as ReportRow[])

      // Patient lookup must not determine whether scans are displayed.
      const ids=[...new Set(preds.map(p=>p.patient_id).filter(Boolean))]
      if(!ids.length){ setPatients(new Map()); return }
      const {data:pat,error:pae}=await supabase.from('patients').select('id,name,age,gender,mrn,notes,created_at').in('id',ids)
      if(pae){ console.error('Patient lookup failed:',pae); setError('Scans loaded, but some patient details could not be loaded.'); return }
      const map=new Map<string,Patient>(); (pat||[]).forEach((p:any)=>map.set(p.id,p as Patient)); setPatients(map)
    } catch(e:any) {
      console.error('Reports load error:',e); setError(e?.message||'Unable to load reports')
    } finally { setLoading(false) }
  },[])

  useEffect(()=>{loadData()},[loadData])

  const generate=async(id:string)=>{
    setGeneratingId(id); setError(null)
    try {
      const {data,error}=await supabase.from('predictions').select('*').eq('id',id).single()
      if(error) throw error
      const p=data as unknown as PredictionResult
      const patient=patients.get(p.patient_id)
      if(!patient) throw new Error('Patient record is unavailable for this scan.')
      await generateReportPDF(p,patient)

      const {data:existing,error:ee}=await supabase.from('reports').select('id,prediction_id,report_type,created_at').eq('prediction_id',id).eq('report_type','full').limit(1).maybeSingle()
      if(ee) throw ee
      if(!existing){
        const {data:n,error:ie}=await supabase.from('reports').insert({prediction_id:id,report_type:'full'}).select('id,prediction_id,report_type,created_at').single()
        if(ie) throw ie
        setReports(x=>[n as ReportRow,...x])
      }
    } catch(e:any) { console.error('Report generation error:',e); setError(e?.message||'Report generation failed') }
    finally { setGeneratingId(null) }
  }

  const filtered=predictions.filter(p=>{
    if(classFilter!=='all'&&p.predicted_class!==classFilter)return false
    if(search.trim()){
      const pt=patients.get(p.patient_id)
      if(!`${pt?.name||''} ${pt?.mrn||''} ${p.predicted_class_display}`.toLowerCase().includes(search.trim().toLowerCase()))return false
    }
    return true
  })
  const reported=new Set(reports.map(r=>r.prediction_id))

  return <div>
    <PageHeader title="Reports" description="Generate and download professional PDF diagnostic reports"/>
    <div className="mb-6 card p-4"><div className="flex flex-wrap items-center gap-3"><div className="relative flex-1 min-w-[200px]"><Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400"/><input value={search} onChange={e=>setSearch(e.target.value)} className="input-field pl-10" placeholder="Search by patient name, MRN, or diagnosis..."/></div><div className="flex items-center gap-2"><Filter className="h-4 w-4 text-neutral-400"/><select value={classFilter} onChange={e=>setClassFilter(e.target.value)} className="input-field w-auto"><option value="all">All Types</option><option value="glioma">Glioma</option><option value="meningioma">Meningioma</option><option value="pituitary">Pituitary</option><option value="no_tumor">No Tumor</option></select></div></div></div>
    {error&&<div className="mb-6 rounded-lg bg-error-50 px-4 py-3 text-sm text-error-700">{error}</div>}
    {reports.length>0&&<div className="mb-8"><h2 className="mb-4 text-lg font-semibold">Generated Reports</h2><div className="card divide-y">{reports.slice(0,10).map(r=>{const p=predictions.find(x=>x.id===r.prediction_id),pt=p?patients.get(p.patient_id):undefined;return <div key={r.id} className="flex items-center justify-between p-4"><div><p className="text-sm font-semibold">{pt?.name||'Patient record unavailable'}</p><p className="text-xs text-neutral-500">{pt?.mrn||'—'} — {p?TUMOR_CLASS_DISPLAY[p.predicted_class]:'—'} — {formatDate(r.created_at)}</p></div>{p&&<div className="flex gap-2"><Link to={`/prediction/${p.id}`} className="btn-secondary text-xs">View</Link><button onClick={()=>generate(p.id)} disabled={generatingId===p.id} className="btn-secondary text-xs">{generatingId===p.id?<Loader2 className="h-3.5 w-3.5 animate-spin"/>:<Download className="h-3.5 w-3.5"/>}Download</button></div>}</div>})}</div></div>}
    <h2 className="mb-4 text-lg font-semibold">Available Scans</h2>
    {loading?<div className="space-y-3">{[1,2,3].map(i=><div key={i} className="h-20 rounded-lg shimmer-bg"/>)}</div>:filtered.length?<div className="space-y-3">{filtered.map(p=>{const pt=patients.get(p.patient_id),has=reported.has(p.id);return <div key={p.id} className="card flex items-center justify-between p-4"><div><p className="text-sm font-semibold">{pt?.name||'Patient record unavailable'}</p><p className="text-xs text-neutral-500">{pt?.mrn||'—'} — {formatDate(p.created_at)}</p><PredictionBadge tumorClass={p.predicted_class} confidence={p.uncertainty?.confidence??0} isUncertain={p.uncertainty?.is_uncertain??false} size="sm"/></div><div className="flex items-center gap-3"><Link to={`/prediction/${p.id}`} className="text-sm text-primary-600">View</Link><button onClick={()=>generate(p.id)} disabled={generatingId===p.id} className="btn-primary text-xs">{generatingId===p.id?<><Loader2 className="h-3.5 w-3.5 animate-spin"/>Generating...</>:<><Download className="h-3.5 w-3.5"/>{has?'Regenerate':'Generate Report'}</>}</button></div></div>})}</div>:<div className="py-20 text-center"><FileText className="mx-auto mb-4 h-12 w-12 text-neutral-400"/><h2 className="text-lg font-semibold">No scans available</h2><p className="mt-1 text-sm text-neutral-400">{predictions.length?'No scans match the selected filters':'Upload an MRI scan to generate diagnostic reports'}</p><Link to="/upload" className="btn-primary mt-4">Upload MRI</Link></div>}
  </div>
}
