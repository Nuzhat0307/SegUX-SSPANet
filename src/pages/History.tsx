import { useState, useEffect, useCallback, useMemo } from 'react'
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
import { PredictionResult, Patient } from '../lib/types'
import PageHeader from '../components/PageHeader'
import PredictionCard from '../components/PredictionCard'

const PAGE_SIZE = 9

export default function History() {
  const [searchParams, setSearchParams] = useSearchParams()

  const [allPredictions, setAllPredictions] = useState<PredictionResult[]>([])
  const [patients, setPatients] = useState<Map<string, Patient>>(new Map())

  const [loading, setLoading] = useState(true)

  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState<string>('all')
  const [uncertainOnly, setUncertainOnly] = useState(false)

  const currentPage = Math.max(
    1,
    parseInt(searchParams.get('page') || '1', 10)
  )

  const patientFilter = searchParams.get('patient') || ''

  // ============================================================
  // LOAD ALL USER PREDICTIONS
  // ============================================================

  const loadPredictions = useCallback(async () => {
    setLoading(true)

    try {
      /*
       * IMPORTANT:
       *
       * Do NOT apply pagination here.
       *
       * The previous implementation used:
       *
       *   .range(from, to)
       *
       * before search/filtering.
       *
       * That caused search to operate only on the current page.
       *
       * We load the user's prediction records first and then
       * perform filtering + pagination on the complete result set.
       */

      let query = supabase
        .from('predictions')
        .select('*')
        .order('created_at', { ascending: false })

      if (patientFilter) {
        query = query.eq('patient_id', patientFilter)
      }

      const { data, error } = await query

      if (error) {
        throw error
      }

      const preds = (data || []) as unknown as PredictionResult[]

      setAllPredictions(preds)

      // ----------------------------------------------------------
      // Load ALL associated patients
      // ----------------------------------------------------------

      if (preds.length > 0) {
        const patientIds = [
          ...new Set(
            preds
              .map((p) => p.patient_id)
              .filter(Boolean)
          ),
        ]

        if (patientIds.length > 0) {
          const { data: patData, error: patientError } =
            await supabase
              .from('patients')
              .select('*')
              .in('id', patientIds)

          if (patientError) {
            throw patientError
          }

          const patMap = new Map<string, Patient>()

          ;(patData || []).forEach((p: any) => {
            patMap.set(
              p.id,
              p as Patient
            )
          })

          setPatients(patMap)
        } else {
          setPatients(new Map())
        }
      } else {
        setPatients(new Map())
      }
    } catch (err) {
      console.error('History load error:', err)

      setAllPredictions([])
      setPatients(new Map())
    } finally {
      setLoading(false)
    }
  }, [patientFilter])

  useEffect(() => {
    loadPredictions()
  }, [loadPredictions])

  // ============================================================
  // FILTER ALL RECORDS
  // ============================================================

  const filteredPredictions = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase()

    return allPredictions.filter((prediction) => {
      // --------------------------------------------------------
      // Class filter
      // --------------------------------------------------------

      if (
        classFilter !== 'all' &&
        prediction.predicted_class !== classFilter
      ) {
        return false
      }

      // --------------------------------------------------------
      // Uncertain-only filter
      // --------------------------------------------------------

      if (
        uncertainOnly &&
        !prediction.uncertainty?.is_uncertain
      ) {
        return false
      }

      // --------------------------------------------------------
      // Search
      //
      // Searches across ALL loaded records:
      //   - Patient name
      //   - MRN
      //   - Diagnosis
      // --------------------------------------------------------

      if (normalizedSearch) {
        const patient = patients.get(
          prediction.patient_id
        )

        const searchText = [
          patient?.name || '',
          patient?.mrn || '',
          prediction.predicted_class || '',
          prediction.predicted_class_display || '',
        ]
          .join(' ')
          .toLowerCase()

        if (!searchText.includes(normalizedSearch)) {
          return false
        }
      }

      return true
    })
  }, [
    allPredictions,
    patients,
    search,
    classFilter,
    uncertainOnly,
  ])

  // ============================================================
  // PAGINATION
  //
  // Pagination happens AFTER filtering.
  // ============================================================

  const totalFilteredCount =
    filteredPredictions.length

  const totalPages = Math.max(
    1,
    Math.ceil(
      totalFilteredCount / PAGE_SIZE
    )
  )

  /*
   * Prevent an invalid page when filtering reduces the
   * number of available pages.
   */
  const safeCurrentPage = Math.min(
    currentPage,
    totalPages
  )

  const paginatedPredictions =
    useMemo(() => {
      const from =
        (safeCurrentPage - 1) *
        PAGE_SIZE

      const to =
        from + PAGE_SIZE

      return filteredPredictions.slice(
        from,
        to
      )
    }, [
      filteredPredictions,
      safeCurrentPage,
    ])

  // ============================================================
  // PAGE NAVIGATION
  // ============================================================

  const goToPage = (page: number) => {
    const targetPage = Math.min(
      Math.max(1, page),
      totalPages
    )

    const params =
      new URLSearchParams(searchParams)

    params.set(
      'page',
      String(targetPage)
    )

    setSearchParams(params)
  }

  // ============================================================
  // RESET TO PAGE 1 WHEN SEARCH/FILTER CHANGES
  // ============================================================

  const resetToFirstPage = () => {
    const params =
      new URLSearchParams(searchParams)

    params.set('page', '1')

    setSearchParams(params)
  }

  const handleSearchChange = (
    value: string
  ) => {
    setSearch(value)
    resetToFirstPage()
  }

  const handleClassFilterChange = (
    value: string
  ) => {
    setClassFilter(value)
    resetToFirstPage()
  }

  const handleUncertainChange = (
    checked: boolean
  ) => {
    setUncertainOnly(checked)
    resetToFirstPage()
  }

  // ============================================================
  // CLEAR FILTERS
  // ============================================================

  const clearFilters = () => {
    setSearch('')
    setClassFilter('all')
    setUncertainOnly(false)

    /*
     * Remove page and patient query parameters.
     */
    setSearchParams({})
  }

  // ============================================================
  // CLASS OPTIONS
  // ============================================================

  const classOptions: {
    value: string
    label: string
  }[] = [
    {
      value: 'all',
      label: 'All Types',
    },
    {
      value: 'glioma',
      label: 'Glioma',
    },
    {
      value: 'meningioma',
      label: 'Meningioma',
    },
    {
      value: 'pituitary',
      label: 'Pituitary',
    },
    {
      value: 'no_tumor',
      label: 'No Tumor',
    },
  ]

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div>
      <PageHeader
        title="Analysis History"
        description="Browse and search through all MRI analyses"
      />

      {/* ======================================================
          FILTERS
          ====================================================== */}

      <div className="mb-6 card p-4">
        <div className="flex flex-wrap items-center gap-3">

          {/* Search */}

          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />

            <input
              type="text"
              value={search}
              onChange={(e) =>
                handleSearchChange(
                  e.target.value
                )
              }
              className="input-field pl-10"
              placeholder="Search by patient name, MRN, or diagnosis..."
            />
          </div>

          {/* Class filter */}

          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-neutral-400" />

            <select
              value={classFilter}
              onChange={(e) =>
                handleClassFilterChange(
                  e.target.value
                )
              }
              className="input-field w-auto"
            >
              {classOptions.map(
                (opt) => (
                  <option
                    key={opt.value}
                    value={opt.value}
                  >
                    {opt.label}
                  </option>
                )
              )}
            </select>
          </div>

          {/* Uncertain only */}

          <label className="flex cursor-pointer items-center gap-2 text-sm font-medium text-neutral-600">

            <input
              type="checkbox"
              checked={uncertainOnly}
              onChange={(e) =>
                handleUncertainChange(
                  e.target.checked
                )
              }
              className="h-4 w-4 rounded border-neutral-300 text-primary-600 focus:ring-primary-500"
            />

            <span className="flex items-center gap-1">
              <AlertTriangle className="h-3.5 w-3.5 text-warning-500" />
              Uncertain only
            </span>

          </label>

          {/* Clear */}

          {(search ||
            classFilter !== 'all' ||
            uncertainOnly ||
            patientFilter) && (
            <button
              onClick={clearFilters}
              className="text-sm font-medium text-primary-600 hover:text-primary-700"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* ======================================================
          RESULTS COUNT
          ====================================================== */}

      <div className="mb-4 flex items-center justify-between">
        <p className="text-sm text-neutral-500">
          {loading
            ? 'Loading...'
            : search ||
                classFilter !== 'all' ||
                uncertainOnly
              ? `${totalFilteredCount} matching analyses`
              : `${totalFilteredCount} analyses`}
        </p>
      </div>

      {/* ======================================================
          RESULTS
          ====================================================== */}

      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map(
            (i) => (
              <div
                key={i}
                className="h-28 rounded-xl shimmer-bg"
              />
            )
          )}
        </div>
      ) : paginatedPredictions.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">

          {paginatedPredictions.map(
            (pred) => (
              <PredictionCard
                key={pred.id}
                prediction={pred}
                patientName={
                  patients.get(
                    pred.patient_id
                  )?.name
                }
              />
            )
          )}

        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 text-center">

          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-neutral-100">
            <Scan className="h-8 w-8 text-neutral-400" />
          </div>

          <h2 className="text-lg font-semibold text-neutral-700">
            No analyses found
          </h2>

          <p className="mt-1 text-sm text-neutral-400">
            {allPredictions.length === 0
              ? "You haven't analyzed any MRI scans yet"
              : 'No results match your filters'}
          </p>

          <Link
            to="/upload"
            className="btn-primary mt-4"
          >
            <Scan className="h-4 w-4" />
            Upload MRI
          </Link>

        </div>
      )}

      {/* ======================================================
          PAGINATION
          ====================================================== */}

      {totalFilteredCount > PAGE_SIZE && (
        <div className="mt-8 flex items-center justify-center gap-2">

          {/* Previous */}

          <button
            onClick={() =>
              goToPage(
                safeCurrentPage - 1
              )
            }
            disabled={
              safeCurrentPage === 1
            }
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 bg-white text-neutral-600 transition-all hover:bg-neutral-50 disabled:opacity-40"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>

          {/* Page numbers */}

          {Array.from(
            {
              length: totalPages,
            },
            (_, i) => i + 1
          )
            .filter(
              (p) =>
                p === 1 ||
                p === totalPages ||
                Math.abs(
                  p - safeCurrentPage
                ) <= 1
            )
            .map(
              (
                page,
                idx,
                arr
              ) => (
                <span
                  key={page}
                  className="flex items-center gap-2"
                >
                  {idx > 0 &&
                    arr[idx - 1] !==
                      page - 1 && (
                      <span className="text-neutral-400">
                        ...
                      </span>
                    )}

                  <button
                    onClick={() =>
                      goToPage(page)
                    }
                    className={`flex h-9 w-9 items-center justify-center rounded-lg text-sm font-medium transition-all ${
                      page ===
                      safeCurrentPage
                        ? 'bg-primary-600 text-white'
                        : 'border border-neutral-200 bg-white text-neutral-600 hover:bg-neutral-50'
                    }`}
                  >
                    {page}
                  </button>
                </span>
              )
            )}

          {/* Next */}

          <button
            onClick={() =>
              goToPage(
                safeCurrentPage + 1
              )
            }
            disabled={
              safeCurrentPage ===
              totalPages
            }
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 bg-white text-neutral-600 transition-all hover:bg-neutral-50 disabled:opacity-40"
          >
            <ChevronRight className="h-4 w-4" />
          </button>

        </div>
      )}
    </div>
  )
}