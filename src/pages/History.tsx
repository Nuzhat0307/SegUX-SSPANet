import { useState, useEffect, useCallback } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import {
  Search,
  Filter,
  Scan,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import { PredictionResult, Patient, TumorClass } from '../lib/types'
import PageHeader from '../components/PageHeader'
import PredictionCard from '../components/PredictionCard'

const PAGE_SIZE = 9

export default function History() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [patients, setPatients] = useState<Map<string, Patient>>(new Map())
  const [loading, setLoading] = useState(true)
  const [totalCount, setTotalCount] = useState(0)
  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState<string>('all')
  const [uncertainOnly, setUncertainOnly] = useState(false)

  const currentPage = parseInt(searchParams.get('page') || '1')
  const patientFilter = searchParams.get('patient') || ''

  const loadPredictions = useCallback(async () => {
    setLoading(true)
    try {
      let query = supabase
        .from('predictions')
        .select('*', { count: 'exact' })
        .order('created_at', { ascending: false })

      if (patientFilter) {
        query = query.eq('patient_id', patientFilter)
      }

      const from = (currentPage - 1) * PAGE_SIZE
      const to = from + PAGE_SIZE - 1
      query = query.range(from, to)

      const { data, count } = await query
      const preds = (data || []) as unknown as PredictionResult[]
      setPredictions(preds)
      setTotalCount(count || 0)

      // Load associated patients
      if (preds.length > 0) {
        const patientIds = [...new Set(preds.map((p) => p.patient_id))]
        const { data: patData } = await supabase
          .from('patients')
          .select('*')
          .in('id', patientIds)
        const patMap = new Map<string, Patient>()
        ;(patData || []).forEach((p: any) => patMap.set(p.id, p as Patient))
        setPatients(patMap)
      }
    } catch (err) {
      console.error('History load error:', err)
    } finally {
      setLoading(false)
    }
  }, [currentPage, patientFilter])

  useEffect(() => {
    loadPredictions()
  }, [loadPredictions])

  // Filter on the client side (for search and class filter)
  const filteredPredictions = predictions.filter((p) => {
    if (classFilter !== 'all' && p.predicted_class !== classFilter) return false
    if (uncertainOnly && !p.uncertainty?.is_uncertain) return false
    if (search) {
      const patient = patients.get(p.patient_id)
      const searchText = `${patient?.name || ''} ${patient?.mrn || ''} ${p.predicted_class_display}`.toLowerCase()
      if (!searchText.includes(search.toLowerCase())) return false
    }
    return true
  })

  const totalPages = Math.ceil(totalCount / PAGE_SIZE)

  const goToPage = (page: number) => {
    const params = new URLSearchParams(searchParams)
    params.set('page', String(page))
    setSearchParams(params)
  }

  const clearFilters = () => {
    setSearch('')
    setClassFilter('all')
    setUncertainOnly(false)
    setSearchParams({})
  }

  const classOptions: { value: string; label: string }[] = [
    { value: 'all', label: 'All Types' },
    { value: 'glioma', label: 'Glioma' },
    { value: 'meningioma', label: 'Meningioma' },
    { value: 'pituitary', label: 'Pituitary' },
    { value: 'no_tumor', label: 'No Tumor' },
  ]

  return (
    <div>
      <PageHeader
        title="Analysis History"
        description="Browse and search through all MRI analyses"
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
          <label className="flex cursor-pointer items-center gap-2 text-sm font-medium text-neutral-600">
            <input
              type="checkbox"
              checked={uncertainOnly}
              onChange={(e) => setUncertainOnly(e.target.checked)}
              className="h-4 w-4 rounded border-neutral-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="flex items-center gap-1">
              <AlertTriangle className="h-3.5 w-3.5 text-warning-500" />
              Uncertain only
            </span>
          </label>
          {(search || classFilter !== 'all' || uncertainOnly || patientFilter) && (
            <button
              onClick={clearFilters}
              className="text-sm font-medium text-primary-600 hover:text-primary-700"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Results count */}
      <div className="mb-4 flex items-center justify-between">
        <p className="text-sm text-neutral-500">
          {loading
            ? 'Loading...'
            : `${filteredPredictions.length} of ${totalCount} analyses`}
        </p>
      </div>

      {/* Predictions grid */}
      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="h-28 rounded-xl shimmer-bg" />
          ))}
        </div>
      ) : filteredPredictions.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPredictions.map((pred) => (
            <PredictionCard
              key={pred.id}
              prediction={pred}
              patientName={patients.get(pred.patient_id)?.name}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-neutral-100">
            <Scan className="h-8 w-8 text-neutral-400" />
          </div>
          <h2 className="text-lg font-semibold text-neutral-700">No analyses found</h2>
          <p className="mt-1 text-sm text-neutral-400">
            {totalCount === 0
              ? "You haven't analyzed any MRI scans yet"
              : 'No results match your filters'}
          </p>
          <Link to="/upload" className="btn-primary mt-4">
            <Scan className="h-4 w-4" />
            Upload MRI
          </Link>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-8 flex items-center justify-center gap-2">
          <button
            onClick={() => goToPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 bg-white text-neutral-600 transition-all hover:bg-neutral-50 disabled:opacity-40"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          {Array.from({ length: totalPages }, (_, i) => i + 1)
            .filter((p) => p === 1 || p === totalPages || Math.abs(p - currentPage) <= 1)
            .map((page, idx, arr) => (
              <span key={page} className="flex items-center gap-2">
                {idx > 0 && arr[idx - 1] !== page - 1 && (
                  <span className="text-neutral-400">...</span>
                )}
                <button
                  onClick={() => goToPage(page)}
                  className={`flex h-9 w-9 items-center justify-center rounded-lg text-sm font-medium transition-all ${
                    page === currentPage
                      ? 'bg-primary-600 text-white'
                      : 'border border-neutral-200 bg-white text-neutral-600 hover:bg-neutral-50'
                  }`}
                >
                  {page}
                </button>
              </span>
            ))}
          <button
            onClick={() => goToPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 bg-white text-neutral-600 transition-all hover:bg-neutral-50 disabled:opacity-40"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  )
}
