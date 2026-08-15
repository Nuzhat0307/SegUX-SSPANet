import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Upload, Users, Scan, TrendingUp, AlertTriangle, Activity, ChevronRight } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../lib/auth'
import PageHeader from '../components/PageHeader'
import PredictionCard from '../components/PredictionCard'
import { PredictionResult, Patient, TumorClass, TUMOR_CLASS_COLORS, TUMOR_CLASS_DISPLAY } from '../lib/types'

interface Stats { total: number; patients: number; uncertain: number; confidence: number }
type Recent = Pick<PredictionResult, 'id' | 'patient_id' | 'predicted_class' | 'predicted_class_display' | 'uncertainty' | 'model_version' | 'inference_time_ms' | 'created_at'>

export default function Dashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState<Stats>({ total: 0, patients: 0, uncertain: 0, confidence: 0 })
  const [recent, setRecent] = useState<Recent[]>([])
  const [patients, setPatients] = useState<Patient[]>([])
  const [distribution, setDistribution] = useState<Record<TumorClass, number>>({ glioma: 0, meningioma: 0, pituitary: 0, no_tumor: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    ;(async () => {
      try {
        const [recentResult, patientResult, patientCountResult, totalResult, gliomaResult, meningiomaResult, pituitaryResult, noTumorResult, uncertaintyResult] = await Promise.all([
          supabase.from('predictions').select('id,patient_id,predicted_class,predicted_class_display,uncertainty,model_version,inference_time_ms,created_at').order('created_at', { ascending: false }).limit(5),
          supabase.from('patients').select('id,name,age,gender,mrn,notes,created_at,predictions(count)').order('created_at', { ascending: false }).limit(5),
          supabase.from('patients').select('id', { count: 'exact', head: true }),
          supabase.from('predictions').select('id', { count: 'exact', head: true }),
          supabase.from('predictions').select('id', { count: 'exact', head: true }).eq('predicted_class', 'glioma'),
          supabase.from('predictions').select('id', { count: 'exact', head: true }).eq('predicted_class', 'meningioma'),
          supabase.from('predictions').select('id', { count: 'exact', head: true }).eq('predicted_class', 'pituitary'),
          supabase.from('predictions').select('id', { count: 'exact', head: true }).eq('predicted_class', 'no_tumor'),
          supabase.from('predictions').select('uncertainty').order('created_at', { ascending: false }).limit(1000),
        ])

        const firstError = [recentResult.error, patientResult.error, patientCountResult.error, totalResult.error, gliomaResult.error, meningiomaResult.error, pituitaryResult.error, noTumorResult.error, uncertaintyResult.error].find(Boolean)
        if (firstError) throw firstError
        if (!active) return

        const uncertaintyRows = (uncertaintyResult.data || []) as any[]
        const uncertain = uncertaintyRows.filter(row => row.uncertainty?.is_uncertain).length
        const confidence = uncertaintyRows.length ? uncertaintyRows.reduce((sum, row) => sum + Number(row.uncertainty?.confidence || 0), 0) / uncertaintyRows.length : 0
        const distribution = {
          glioma: gliomaResult.count || 0,
          meningioma: meningiomaResult.count || 0,
          pituitary: pituitaryResult.count || 0,
          no_tumor: noTumorResult.count || 0,
        }
        const mappedPatients = ((patientResult.data || []) as any[]).map(patient => ({ ...patient, prediction_count: patient.predictions?.[0]?.count || 0 }))

        setRecent((recentResult.data || []) as Recent[])
        setPatients(mappedPatients)
        setDistribution(distribution)
        setStats({ total: totalResult.count || 0, patients: patientCountResult.count || 0, uncertain, confidence })
      } catch (error) {
        console.error('Dashboard load error', error)
      } finally {
        if (active) setLoading(false)
      }
    })()
    return () => { active = false }
  }, [])

  const pieStyle = useMemo(() => {
    const entries = Object.entries(distribution) as [TumorClass, number][]
    const total = entries.reduce((sum, [, count]) => sum + count, 0)
    if (!total) return { background: 'conic-gradient(#e5e7eb 0deg 360deg)' }
    let start = 0
    const stops = entries.map(([key, count]) => {
      const end = start + (count / total) * 360
      const stop = `${TUMOR_CLASS_COLORS[key]} ${start}deg ${end}deg`
      start = end
      return stop
    })
    return { background: `conic-gradient(${stops.join(', ')})` }
  }, [distribution])

  const cards = [
    ['Total Scans Analyzed', stats.total, Scan, 'bg-primary-100', 'text-primary-600'],
    ['Total Patients', stats.patients, Users, 'bg-secondary-100', 'text-secondary-600'],
    ['Cases Needing Review', stats.uncertain, AlertTriangle, 'bg-warning-100', 'text-warning-600'],
    ['Average Confidence', `${(stats.confidence * 100).toFixed(1)}%`, TrendingUp, 'bg-success-100', 'text-success-600'],
  ] as const

  return <div>
    <PageHeader title="Dashboard" description={`Welcome back${user?.user_metadata?.full_name ? `, ${user.user_metadata.full_name}` : ''}. Here's your diagnostic overview.`} action={<Link to="/upload" className="btn-primary"><Upload className="h-4 w-4" />New Scan</Link>} />
    <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">{cards.map(([label, value, Icon, bg, color]) => <div key={label} className="card p-5"><div className="flex items-center justify-between"><div><p className="text-sm text-neutral-500">{label}</p><p className="mt-1 text-2xl font-bold">{value}</p></div><div className={`flex h-12 w-12 items-center justify-center rounded-xl ${bg}`}><Icon className={`h-6 w-6 ${color}`} /></div></div></div>)}</div>
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2 card p-5"><div className="mb-4 flex justify-between"><h2 className="text-lg font-semibold">Recent Analyses</h2><Link to="/history" className="flex items-center text-sm text-primary-600">View all<ChevronRight className="h-4 w-4" /></Link></div>{loading ? <div className="space-y-3">{[1, 2, 3].map(i => <div key={i} className="h-24 rounded-lg shimmer-bg" />)}</div> : recent.length ? <div className="space-y-3">{recent.map(p => <PredictionCard key={p.id} prediction={p as PredictionResult} />)}</div> : <div className="py-12 text-center"><Activity className="mx-auto mb-3 h-10 w-10 text-neutral-400" /><p>No analyses yet</p><Link to="/upload" className="btn-primary mt-4"><Upload className="h-4 w-4" />Upload MRI</Link></div>}</div>
      <div className="space-y-6">
        <div className="card p-5"><h2 className="mb-5 text-lg font-semibold">Tumor Distribution</h2><div className="flex items-center gap-6"><div className="relative h-40 w-40 shrink-0 rounded-full" style={pieStyle} aria-label="Tumor distribution pie chart"><div className="absolute inset-8 rounded-full bg-white" /></div><div className="min-w-0 flex-1 space-y-3">{(Object.keys(distribution) as TumorClass[]).map(key => { const count = distribution[key]; const total = Object.values(distribution).reduce((sum, value) => sum + value, 0); const percentage = total ? (count / total) * 100 : 0; return <div key={key} className="flex items-center justify-between gap-3 text-sm"><div className="flex min-w-0 items-center gap-2"><span className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: TUMOR_CLASS_COLORS[key] }} /><span className="truncate">{TUMOR_CLASS_DISPLAY[key]}</span></div><span className="shrink-0 text-neutral-500">{count} ({percentage.toFixed(0)}%)</span></div> })}</div></div></div>
        <div className="card p-5"><h2 className="mb-4 text-lg font-semibold">Recent Patients</h2>{patients.length ? patients.map(p => <Link key={p.id} to={`/history?patient=${p.id}`} className="flex items-center justify-between rounded-lg px-3 py-2.5 hover:bg-neutral-50"><div><p className="text-sm font-medium">{p.name}</p><p className="text-xs text-neutral-400">MRN: {p.mrn}</p></div><span className="badge bg-primary-50 text-primary-600">{p.prediction_count || 0} scans</span></Link>) : <p className="py-8 text-center text-sm text-neutral-400">No patients registered yet</p>}</div>
      </div>
    </div>
  </div>
}
