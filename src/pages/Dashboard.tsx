import { useState,useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Upload,Users,Scan,TrendingUp,AlertTriangle,Activity,ChevronRight } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../lib/auth'
import PageHeader from '../components/PageHeader'
import PredictionCard from '../components/PredictionCard'
import { PredictionResult,Patient,TumorClass,TUMOR_CLASS_COLORS,TUMOR_CLASS_DISPLAY } from '../lib/types'

interface Stats { total:number; patients:number; uncertain:number; confidence:number }
type Recent=Pick<PredictionResult,'id'|'patient_id'|'predicted_class'|'predicted_class_display'|'uncertainty'|'model_version'|'inference_time_ms'|'created_at'>

export default function Dashboard(){
 const {user}=useAuth(); const [stats,setStats]=useState<Stats>({total:0,patients:0,uncertain:0,confidence:0}); const [recent,setRecent]=useState<Recent[]>([]); const [patients,setPatients]=useState<Patient[]>([]); const [distribution,setDistribution]=useState<Record<string,number>>({glioma:0,meningioma:0,pituitary:0,no_tumor:0}); const [loading,setLoading]=useState(true)
 useEffect(()=>{let active=true;(async()=>{setLoading(true);try{
   const [{data:recentData,error:re},{data:patientData,error:pe},{data:summary,error:se}]=await Promise.all([
    supabase.from('predictions').select('id,patient_id,predicted_class,predicted_class_display,uncertainty,model_version,inference_time_ms,created_at').order('created_at',{ascending:false}).limit(5),
    supabase.from('patients').select('id,name,age,gender,mrn,notes,created_at').order('created_at',{ascending:false}).limit(5),
    supabase.from('predictions').select('patient_id,predicted_class,uncertainty')
   ]); if(re)throw re;if(pe)throw pe;if(se)throw se;if(!active)return
   const rows=(summary||[]) as any[];const dist={glioma:0,meningioma:0,pituitary:0,no_tumor:0} as Record<string,number>;let uncertain=0,conf=0
   rows.forEach(p=>{if(dist[p.predicted_class]!==undefined)dist[p.predicted_class]++;if(p.uncertainty?.is_uncertain)uncertain++;conf+=Number(p.uncertainty?.confidence||0)})
   const patientCounts=new Map<string,number>();rows.forEach(p=>patientCounts.set(p.patient_id,(patientCounts.get(p.patient_id)||0)+1));const pts=(patientData||[]).map((p:any)=>({...p,prediction_count:patientCounts.get(p.id)||0})) as Patient[]
   setRecent((recentData||[]) as Recent[]);setPatients(pts);setDistribution(dist);setStats({total:rows.length,patients:(patientData||[]).length,uncertain,confidence:rows.length?conf/rows.length:0})
  }catch(e){console.error('Dashboard load error',e)}finally{if(active)setLoading(false)}})();return()=>{active=false}},[])
 const cards=[['Total Scans Analyzed',stats.total,Scan,'bg-primary-100','text-primary-600'],['Total Patients',stats.patients,Users,'bg-secondary-100','text-secondary-600'],['Cases Needing Review',stats.uncertain,AlertTriangle,'bg-warning-100','text-warning-600'],['Average Confidence',`${(stats.confidence*100).toFixed(1)}%`,TrendingUp,'bg-success-100','text-success-600']] as const
 return <div><PageHeader title="Dashboard" description={`Welcome back${user?.user_metadata?.full_name?`, ${user.user_metadata.full_name}`:''}. Here's your diagnostic overview.`} action={<Link to="/upload" className="btn-primary"><Upload className="h-4 w-4"/>New Scan</Link>}/>
 <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">{cards.map(([label,value,Icon,bg,fg])=><div key={label} className="card p-5"><div className="flex items-center justify-between"><div><p className="text-sm font-medium text-neutral-500">{label}</p><p className="mt-1 text-2xl font-bold text-neutral-900">{value}</p></div><div className={`flex h-12 w-12 items-center justify-center rounded-xl ${bg}`}><Icon className={`h-6 w-6 ${fg}`}/></div></div></div>)}</div>
 <div className="grid grid-cols-1 gap-6 lg:grid-cols-3"><div className="lg:col-span-2"><div className="card p-5"><div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-semibold text-neutral-900">Recent Analyses</h2><Link to="/history" className="flex items-center gap-1 text-sm font-medium text-primary-600">View all<ChevronRight className="h-4 w-4"/></Link></div>{loading?<div className="space-y-3">{[1,2,3].map(i=><div key={i} className="h-24 rounded-lg shimmer-bg"/>)}</div>:recent.length?<div className="space-y-3">{recent.map(p=><PredictionCard key={p.id} prediction={p as PredictionResult}/>)}</div>:<div className="flex flex-col items-center py-12 text-center"><Activity className="mb-3 h-10 w-10 text-neutral-400"/><p className="text-sm text-neutral-600">No analyses yet</p><Link to="/upload" className="btn-primary mt-4"><Upload className="h-4 w-4"/>Upload MRI</Link></div>}</div></div>
 <div className="space-y-6"><div className="card p-5"><h2 className="mb-4 text-lg font-semibold text-neutral-900">Tumor Distribution</h2><div className="space-y-3">{Object.entries(distribution).map(([key,count])=>{const total=Math.max(1,Object.values(distribution).reduce((a,b)=>a+b,0));return <div key={key}><div className="mb-1 flex justify-between text-xs"><span>{TUMOR_CLASS_DISPLAY[key as TumorClass]}</span><span>{count}</span></div><div className="h-2 overflow-hidden rounded-full bg-neutral-100"><div className="h-full rounded-full" style={{width:`${count/total*100}%`,backgroundColor:TUMOR_CLASS_COLORS[key as TumorClass]}}/></div></div>})}</div></div>
 <div className="card p-5"><h2 className="mb-4 text-lg font-semibold text-neutral-900">Recent Patients</h2>{patients.length?<div className="space-y-2">{patients.map((p:any)=><Link key={p.id} to={`/history?patient=${p.id}`} className="flex items-center justify-between rounded-lg px-3 py-2.5 hover:bg-neutral-50"><div><p className="text-sm font-medium text-neutral-900">{p.name}</p><p className="text-xs text-neutral-400">MRN: {p.mrn}</p></div><span className="badge bg-primary-50 text-primary-600">{p.prediction_count} scans</span></Link>)}</div>:<p className="py-8 text-center text-sm text-neutral-400">No patients registered yet</p>}</div></div></div></div>
}
