import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Upload,
  Users,
  Scan,
  TrendingUp,
  AlertTriangle,
  Activity,
  Brain,
  ChevronRight,
  Clock,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../lib/auth'
import PageHeader from '../components/PageHeader'
import PredictionCard from '../components/PredictionCard'
import { PredictionResult, Patient, TumorClass, TUMOR_CLASS_COLORS } from '../lib/types'
import { formatDate } from '../lib/utils'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'

interface DashboardStats {
  totalPredictions: number
  totalPatients: number
  uncertainCases: number
  avgConfidence: number
}

interface PatientWithCount extends Patient {
  prediction_count: number
  predictions?: { created_at: string; predicted_class: TumorClass }[]
}

export default function Dashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState<DashboardStats>({
    totalPredictions: 0,
    totalPatients: 0,
    uncertainCases: 0,
    avgConfidence: 0,
  })
  const [recentPredictions, setRecentPredictions] = useState<PredictionResult[]>([])
  const [patients, setPatients] = useState<PatientWithCount[]>([])
  const [classDistribution, setClassDistribution] = useState<{ name: string; value: number; color: string }[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadData() {
      setLoading(true)
      try {
        const [
          { data: predictionsData },
          { data: patientsData },
        ] = await Promise.all([
          supabase
            .from('predictions')
            .select('*')
            .order('created_at', { ascending: false })
            .limit(10),
          supabase
            .from('patients')
            .select('*')
            .order('created_at', { ascending: false }),
        ])

        const allPredictions = (predictionsData || []) as unknown as PredictionResult[]
        const allPatients = (patientsData || []) as unknown as Patient[]

        // Calculate stats
        const uncertainCount = allPredictions.filter(
          (p) => p.uncertainty?.is_uncertain,
        ).length
        const avgConf =
          allPredictions.length > 0
            ? allPredictions.reduce((sum, p) => sum + (p.uncertainty?.confidence || 0), 0) /
              allPredictions.length
            : 0

        // Get full prediction count per patient
        const { count: totalPredCount } = await supabase
          .from('predictions')
          .select('*', { count: 'exact', head: true })

        const patientsWithCounts = await Promise.all(
          (allPatients || []).slice(0, 5).map(async (p) => {
            const { count } = await supabase
              .from('predictions')
              .select('*', { count: 'exact', head: true })
              .eq('patient_id', p.id)
            return { ...p, prediction_count: count || 0 }
          }),
        )

        setRecentPredictions(allPredictions.slice(0, 5))
        setPatients(patientsWithCounts)
        setStats({
          totalPredictions: totalPredCount || 0,
          totalPatients: allPatients.length,
          uncertainCases: uncertainCount,
          avgConfidence: avgConf,
        })

        // Class distribution
        const classCounts: Record<string, number> = {
          glioma: 0,
          meningioma: 0,
          pituitary: 0,
          no_tumor: 0,
        }
        // Fetch ALL predictions for distribution (up to a reasonable limit)
        const { data: allPreds } = await supabase
          .from('predictions')
          .select('predicted_class')
          .limit(1000)
        ;(allPreds || []).forEach((p: any) => {
          if (classCounts[p.predicted_class] !== undefined) {
            classCounts[p.predicted_class]++
          }
        })
        const dist = Object.entries(classCounts).map(([key, value]) => ({
          name: key.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
          value,
          color: TUMOR_CLASS_COLORS[key as TumorClass],
        }))
        setClassDistribution(dist)
      } catch (err) {
        console.error('Dashboard load error:', err)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const statCards = [
    {
      label: 'Total Scans Analyzed',
      value: stats.totalPredictions,
      icon: Scan,
      color: 'primary',
      iconBg: 'bg-primary-100',
      iconColor: 'text-primary-600',
    },
    {
      label: 'Total Patients',
      value: stats.totalPatients,
      icon: Users,
      color: 'secondary',
      iconBg: 'bg-secondary-100',
      iconColor: 'text-secondary-600',
    },
    {
      label: 'Cases Needing Review',
      value: stats.uncertainCases,
      icon: AlertTriangle,
      color: 'warning',
      iconBg: 'bg-warning-100',
      iconColor: 'text-warning-600',
    },
    {
      label: 'Average Confidence',
      value: `${(stats.avgConfidence * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: 'success',
      iconBg: 'bg-success-100',
      iconColor: 'text-success-600',
    },
  ]

  return (
    <div>
      <PageHeader
        title="Dashboard"
        description={`Welcome back${user?.user_metadata?.full_name ? `, ${user.user_metadata.full_name}` : ''}. Here's your diagnostic overview.`}
        action={
          <Link to="/upload" className="btn-primary">
            <Upload className="h-4 w-4" />
            New Scan
          </Link>
        }
      />

      {/* Stats grid */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="card animate-slide-up p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-neutral-500">{stat.label}</p>
                  <p className="mt-1 text-2xl font-bold text-neutral-900">{stat.value}</p>
                </div>
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${stat.iconBg}`}>
                  <Icon className={`h-6 w-6 ${stat.iconColor}`} />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Recent predictions */}
        <div className="lg:col-span-2">
          <div className="card p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-neutral-900">Recent Analyses</h2>
              <Link
                to="/history"
                className="flex items-center gap-1 text-sm font-medium text-primary-600 hover:text-primary-700"
              >
                View all <ChevronRight className="h-4 w-4" />
              </Link>
            </div>

            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-24 rounded-lg shimmer-bg" />
                ))}
              </div>
            ) : recentPredictions.length > 0 ? (
              <div className="space-y-3">
                {recentPredictions.map((pred) => (
                  <PredictionCard key={pred.id} prediction={pred} />
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-neutral-100">
                  <Activity className="h-7 w-7 text-neutral-400" />
                </div>
                <p className="text-sm font-medium text-neutral-600">No analyses yet</p>
                <p className="mt-1 text-xs text-neutral-400">
                  Upload an MRI scan to get started
                </p>
                <Link to="/upload" className="btn-primary mt-4">
                  <Upload className="h-4 w-4" />
                  Upload MRI
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Right column: class distribution + patients */}
        <div className="space-y-6">
          <div className="card p-5">
            <h2 className="mb-4 text-lg font-semibold text-neutral-900">Tumor Distribution</h2>
            {classDistribution.every((d) => d.value === 0) ? (
              <div className="flex h-48 items-center justify-center text-sm text-neutral-400">
                No data yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={classDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={75}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {classDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="card p-5">
            <h2 className="mb-4 text-lg font-semibold text-neutral-900">Recent Patients</h2>
            {patients.length > 0 ? (
              <div className="space-y-2">
                {patients.map((patient) => (
                  <Link
                    key={patient.id}
                    to={`/history?patient=${patient.id}`}
                    className="flex items-center justify-between rounded-lg px-3 py-2.5 transition-colors hover:bg-neutral-50"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-100">
                        <span className="text-xs font-semibold text-primary-700">
                          {patient.name.charAt(0).toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-neutral-900">{patient.name}</p>
                        <p className="text-xs text-neutral-400">MRN: {patient.mrn}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="badge bg-primary-50 text-primary-600">
                        {patient.prediction_count} scans
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-neutral-100">
                  <Users className="h-6 w-6 text-neutral-400" />
                </div>
                <p className="text-sm text-neutral-400">No patients registered yet</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
