import { useState, useRef, useCallback, FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileImage, X, Loader2, Brain, CheckCircle, User, AlertCircle } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { checkBackendHealth, runInference } from '../lib/mockInference'
import { PredictionResult, Patient } from '../lib/types'
import PageHeader from '../components/PageHeader'

type Step = 'select' | 'patient' | 'analyzing' | 'error'

interface PatientForm {
  name: string
  age: string
  gender: 'male' | 'female' | 'other' | ''
  mrn: string
  notes: string
  isNew: boolean
  existingId: string
}

export default function UploadMRI() {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [step, setStep] = useState<Step>('select')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [progressLabel, setProgressLabel] = useState('')
  const [existingPatients, setExistingPatients] = useState<Patient[]>([])
  const [patientForm, setPatientForm] = useState<PatientForm>({ name: '', age: '', gender: '', mrn: '', notes: '', isNew: true, existingId: '' })

  const loadExistingPatients = useCallback(async () => {
    const { data } = await supabase.from('patients').select('*').order('created_at', { ascending: false }).limit(50)
    setExistingPatients((data || []) as unknown as Patient[])
  }, [])

  const handleFile = (selectedFile: File | null) => {
    if (!selectedFile) return
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, or DICOM-converted image)')
      return
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB')
      return
    }
    setError(null)
    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setStep('patient')
    loadExistingPatients()
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true)
    else if (e.type === 'dragleave') setDragActive(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
  }, [])

  const resetUpload = () => {
    setFile(null); setPreviewUrl(null); setStep('select'); setError(null); setProgress(0); setProgressLabel('')
    setPatientForm({ name: '', age: '', gender: '', mrn: '', notes: '', isNew: true, existingId: '' })
  }

  const runAnalysis = async () => {
    if (!file) return

    if (patientForm.isNew && (!patientForm.name || !patientForm.mrn)) {
      setError('Patient name and MRN are required for new patients')
      return
    }
    if (!patientForm.isNew && !patientForm.existingId) {
      setError('Please select a patient')
      return
    }

    setError(null)
    setStep('analyzing')
    setProgress(5)
    setProgressLabel('Checking AI backend and trained model...')

    try {
      // Do this BEFORE creating a patient record so an offline backend cannot
      // leave behind an orphan patient when prediction cannot run.
      await checkBackendHealth()
      setProgress(15)
      setProgressLabel('Submitting MRI to the trained SegUX-SSPANet model...')

      let patientId = patientForm.existingId

      if (patientForm.isNew) {
        const { data: newPatient, error: patientErr } = await supabase
          .from('patients')
          .insert({
            name: patientForm.name,
            age: patientForm.age ? parseInt(patientForm.age) : null,
            gender: patientForm.gender || null,
            mrn: patientForm.mrn,
            notes: patientForm.notes || null,
          })
          .select()
          .single()
        if (patientErr) throw new Error(patientErr.message)
        patientId = (newPatient as any).id
      }

      if (!patientId) throw new Error('No patient selected')

      setProgress(20)
      setProgressLabel('AI model is processing classification, segmentation, explainability and uncertainty...')

      // This calls the real FastAPI /predict endpoint. There is no mock inference.
      const result: PredictionResult = await runInference(file, patientId)
      setProgress(95)
      setProgressLabel('Saving analysis results...')

      const { error: predErr } = await supabase.from('predictions').insert({
        id: result.id,
        patient_id: patientId,
        image_base64: result.image_base64,
        predicted_class: result.predicted_class,
        predicted_class_display: result.predicted_class_display,
        probabilities: result.probabilities,
        uncertainty: result.uncertainty,
        segmentation: result.segmentation,
        gradcam_results: result.gradcam_results,
        feature_explanation: result.feature_explanation,
        model_version: result.model_version,
        inference_time_ms: result.inference_time_ms,
        notes: result.notes,
      })

      if (predErr) throw new Error(predErr.message)

      setProgress(100)
      setProgressLabel('Analysis complete')
      navigate(`/prediction/${result.id}`)
    } catch (err: any) {
      setError(err?.message || 'Analysis failed. Please try again.')
      setStep('error')
    }
  }

  const handlePatientSubmit = (e: FormEvent) => { e.preventDefault(); runAnalysis() }

  return (
    <div>
      <PageHeader title="Upload MRI Scan" description="Upload a brain MRI image for AI-powered tumor analysis with segmentation, explainability, and uncertainty estimation" />

      {step === 'select' && (
        <div className="mx-auto max-w-2xl">
          <div onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()} className={`card flex cursor-pointer flex-col items-center justify-center px-6 py-16 transition-all ${dragActive ? 'border-primary-500 bg-primary-50' : 'hover:border-primary-300 hover:bg-primary-50/50'}`}>
            <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-primary-100"><Upload className="h-10 w-10 text-primary-600" /></div>
            <h3 className="text-lg font-semibold text-neutral-900">Drop your MRI scan here</h3>
            <p className="mt-1 text-sm text-neutral-500">or click to browse files</p>
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-neutral-400"><span className="badge bg-neutral-100 text-neutral-600">JPG</span><span className="badge bg-neutral-100 text-neutral-600">PNG</span><span className="badge bg-neutral-100 text-neutral-600">Max 10MB</span></div>
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files?.[0] || null)} />
          </div>
          {error && <div className="mt-4 flex items-center gap-2 rounded-lg bg-error-50 px-4 py-3 text-sm text-error-700"><AlertCircle className="h-4 w-4 flex-shrink-0" /><span>{error}</span></div>}
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">{[
            { icon: Brain, title: 'Classification', desc: 'SSPANet + ResNet50' },
            { icon: FileImage, title: 'Segmentation', desc: 'U-Net tumor masking' },
            { icon: CheckCircle, title: 'Explainability', desc: 'GradCAM + uncertainty' },
          ].map((f) => { const Icon = f.icon; return <div key={f.title} className="card p-4"><Icon className="mb-2 h-6 w-6 text-primary-600" /><p className="text-sm font-semibold text-neutral-900">{f.title}</p><p className="text-xs text-neutral-500">{f.desc}</p></div> })}</div>
        </div>
      )}

      {step === 'patient' && previewUrl && (
        <form onSubmit={handlePatientSubmit} className="mx-auto max-w-3xl space-y-6">
          <div className="card overflow-hidden"><div className="flex items-center justify-between border-b border-neutral-200 bg-neutral-50 px-5 py-3"><div className="flex items-center gap-2 text-sm font-medium text-neutral-700"><FileImage className="h-4 w-4 text-primary-600" />Selected MRI Scan</div><button type="button" onClick={resetUpload} className="flex items-center gap-1 text-sm text-neutral-500 hover:text-error-600"><X className="h-4 w-4" /> Remove</button></div><div className="flex justify-center bg-neutral-900 p-6"><img src={previewUrl} alt="MRI preview" className="max-h-80 rounded-lg object-contain" /></div>{file && <div className="border-t border-neutral-200 px-5 py-3 text-sm text-neutral-500"><span className="font-medium text-neutral-700">{file.name}</span>{' — '}{(file.size / 1024 / 1024).toFixed(2)} MB</div>}</div>
          <div className="card p-6"><h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-neutral-900"><User className="h-5 w-5 text-primary-600" />Patient Information</h3>
            {existingPatients.length > 0 && <div className="mb-5 flex gap-2 rounded-lg bg-neutral-100 p-1"><button type="button" onClick={() => setPatientForm({ ...patientForm, isNew: true })} className={`flex-1 rounded-md py-2 text-sm font-medium transition-all ${patientForm.isNew ? 'bg-white text-primary-700 shadow-sm' : 'text-neutral-500'}`}>New Patient</button><button type="button" onClick={() => setPatientForm({ ...patientForm, isNew: false })} className={`flex-1 rounded-md py-2 text-sm font-medium transition-all ${!patientForm.isNew ? 'bg-white text-primary-700 shadow-sm' : 'text-neutral-500'}`}>Existing Patient</button></div>}
            {patientForm.isNew ? <div className="grid grid-cols-1 gap-4 sm:grid-cols-2"><div><label className="label-text">Patient Name *</label><input type="text" value={patientForm.name} onChange={(e) => setPatientForm({ ...patientForm, name: e.target.value })} className="input-field" placeholder="John Doe" required /></div><div><label className="label-text">Medical Record Number *</label><input type="text" value={patientForm.mrn} onChange={(e) => setPatientForm({ ...patientForm, mrn: e.target.value })} className="input-field" placeholder="MRN-001" required /></div><div><label className="label-text">Age</label><input type="number" value={patientForm.age} onChange={(e) => setPatientForm({ ...patientForm, age: e.target.value })} className="input-field" placeholder="45" min="0" max="120" /></div><div><label className="label-text">Gender</label><select value={patientForm.gender} onChange={(e) => setPatientForm({ ...patientForm, gender: e.target.value as any })} className="input-field"><option value="">Select...</option><option value="male">Male</option><option value="female">Female</option><option value="other">Other</option></select></div><div className="sm:col-span-2"><label className="label-text">Clinical Notes (optional)</label><textarea value={patientForm.notes} onChange={(e) => setPatientForm({ ...patientForm, notes: e.target.value })} className="input-field min-h-[80px] resize-y" placeholder="Relevant clinical history, symptoms, prior treatments..." /></div></div> : <div><label className="label-text">Select Patient</label><select value={patientForm.existingId} onChange={(e) => setPatientForm({ ...patientForm, existingId: e.target.value })} className="input-field" required><option value="">Select a patient...</option>{existingPatients.map((p) => <option key={p.id} value={p.id}>{p.name} — MRN: {p.mrn}</option>)}</select></div>}
          </div>
          {error && <div className="flex items-center gap-2 rounded-lg bg-error-50 px-4 py-3 text-sm text-error-700"><AlertCircle className="h-4 w-4 flex-shrink-0" /><span>{error}</span></div>}
          <div className="flex justify-end gap-3"><button type="button" onClick={resetUpload} className="btn-secondary">Cancel</button><button type="submit" className="btn-primary"><Brain className="h-4 w-4" />Run Analysis</button></div>
        </form>
      )}

      {step === 'analyzing' && <div className="mx-auto max-w-lg"><div className="card p-8"><div className="mb-6 flex flex-col items-center text-center"><div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-primary-100"><Brain className="h-10 w-10 animate-pulse-slow text-primary-600" /></div><h3 className="text-lg font-semibold text-neutral-900">Analyzing MRI Scan</h3><p className="mt-1 text-sm text-neutral-500">{progressLabel}</p></div><div className="mb-3 h-2 overflow-hidden rounded-full bg-neutral-100"><div className="h-full rounded-full bg-primary-600 transition-all duration-500 ease-out" style={{ width: `${progress}%` }} /></div><p className="text-center text-xs text-neutral-400">{progress}%</p><p className="mt-5 text-center text-xs text-neutral-500">The trained model is running on the backend. Classification, U-Net segmentation, GradCAM and MC-Dropout are completed by the server before results are returned.</p><div className="mt-6 space-y-2"><div className="flex items-center gap-2 text-xs text-neutral-600"><Loader2 className="h-3.5 w-3.5 animate-spin" />AI inference in progress</div><div className="flex items-center gap-2 text-xs text-neutral-500"><CheckCircle className="h-3.5 w-3.5 text-success-500" />No mock prediction is used</div></div></div></div>}

      {step === 'error' && <div className="mx-auto max-w-lg"><div className="card p-8 text-center"><div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-error-100"><AlertCircle className="h-8 w-8 text-error-600" /></div><h3 className="text-lg font-semibold text-neutral-900">Analysis Failed</h3><p className="mt-2 text-sm text-neutral-500">{error}</p><button onClick={() => { setStep('patient'); setError(null) }} className="btn-primary mt-6">Try Again</button></div></div>}
    </div>
  )
}
