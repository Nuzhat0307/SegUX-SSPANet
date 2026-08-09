import { useState, FormEvent } from 'react'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'
import {
  User,
  Mail,
  Settings as SettingsIcon,
  Shield,
  Bell,
  Info,
  Save,
  Loader2,
  CheckCircle,
  Brain,
  Cpu,
  Layers,
} from 'lucide-react'
import PageHeader from '../components/PageHeader'

export default function Settings() {
  const { user } = useAuth()
  const [fullName, setFullName] = useState(user?.user_metadata?.full_name || '')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [notifications, setNotifications] = useState({
    uncertainCases: true,
    newReports: true,
    modelUpdates: false,
  })

  const handleSaveProfile = async (e: FormEvent) => {
    e.preventDefault()
    setSaving(true)
    const { error } = await supabase.auth.updateUser({
      data: { full_name: fullName },
    })
    setSaving(false)
    if (!error) {
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    }
  }

  const modelInfo = [
    { icon: Brain, label: 'Classification Model', value: 'SSPANet + ResNet50' },
    { icon: Layers, label: 'Segmentation Model', value: 'U-Net' },
    { icon: Cpu, label: 'Uncertainty Method', value: 'Monte Carlo Dropout (30 samples)' },
    { icon: Info, label: 'Model Version', value: 'SegUX-SSPANet-v1.0.0' },
    { icon: Shield, label: 'Explainability', value: 'GradCAM, GradCAM++, EigenGradCAM' },
  ]

  return (
    <div>
      <PageHeader title="Settings" description="Manage your account and system configuration" />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Profile section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Profile */}
          <div className="card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-neutral-900">
              <User className="h-5 w-5 text-primary-600" />
              Profile Information
            </h3>
            <form onSubmit={handleSaveProfile} className="space-y-4">
              <div>
                <label className="label-text">Full Name</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="input-field"
                  placeholder="Dr. Jane Smith"
                />
              </div>
              <div>
                <label className="label-text">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
                  <input
                    type="email"
                    value={user?.email || ''}
                    disabled
                    className="input-field pl-10 bg-neutral-50 text-neutral-500"
                  />
                </div>
                <p className="mt-1 text-xs text-neutral-400">Email cannot be changed</p>
              </div>
              <div className="flex items-center gap-3">
                <button type="submit" disabled={saving} className="btn-primary">
                  {saving ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="h-4 w-4" />
                      Save Changes
                    </>
                  )}
                </button>
                {saved && (
                  <span className="flex items-center gap-1.5 text-sm font-medium text-success-600">
                    <CheckCircle className="h-4 w-4" />
                    Saved successfully
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* Notifications */}
          <div className="card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-neutral-900">
              <Bell className="h-5 w-5 text-primary-600" />
              Notification Preferences
            </h3>
            <div className="space-y-3">
              {[
                {
                  key: 'uncertainCases' as const,
                  label: 'Uncertain case alerts',
                  desc: 'Get notified when a scan is flagged for expert review',
                },
                {
                  key: 'newReports' as const,
                  label: 'Report generation',
                  desc: 'Notify when a new diagnostic report is generated',
                },
                {
                  key: 'modelUpdates' as const,
                  label: 'Model updates',
                  desc: 'Get notified about AI model improvements and updates',
                },
              ].map((item) => (
                <label
                  key={item.key}
                  className="flex cursor-pointer items-center justify-between rounded-lg border border-neutral-200 p-3 transition-colors hover:bg-neutral-50"
                >
                  <div>
                    <p className="text-sm font-medium text-neutral-900">{item.label}</p>
                    <p className="text-xs text-neutral-500">{item.desc}</p>
                  </div>
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={notifications[item.key]}
                      onChange={(e) =>
                        setNotifications({ ...notifications, [item.key]: e.target.checked })
                      }
                      className="peer sr-only"
                    />
                    <div className="h-6 w-11 rounded-full bg-neutral-200 transition-colors peer-checked:bg-primary-600" />
                    <div className="absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-5" />
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Right column: model info */}
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-neutral-900">
              <SettingsIcon className="h-5 w-5 text-primary-600" />
              Model Configuration
            </h3>
            <div className="space-y-3">
              {modelInfo.map((item) => {
                const Icon = item.icon
                return (
                  <div key={item.label} className="flex items-start gap-3">
                    <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-primary-50">
                      <Icon className="h-4 w-4 text-primary-600" />
                    </div>
                    <div>
                      <p className="text-xs font-medium text-neutral-500">{item.label}</p>
                      <p className="text-sm font-semibold text-neutral-900">{item.value}</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card p-6">
            <h3 className="mb-3 flex items-center gap-2 text-lg font-semibold text-neutral-900">
              <Info className="h-5 w-5 text-primary-600" />
              About
            </h3>
            <p className="text-sm text-neutral-600">
              SegUX-SSPANet is a research-grade brain tumor diagnosis system combining
              segmentation-guided attention learning with uncertainty-aware deep learning.
            </p>
            <div className="mt-4 rounded-lg bg-primary-50 p-3">
              <p className="text-xs text-primary-800">
                This system is for research and educational purposes only and is not a substitute
                for professional medical diagnosis by a qualified radiologist or neurologist.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
