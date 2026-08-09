import { TumorClass, TUMOR_CLASS_COLORS, TUMOR_CLASS_DISPLAY } from '../lib/types'
import { getUncertaintyLabel, formatConfidence } from '../lib/utils'
import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'

interface PredictionBadgeProps {
  tumorClass: TumorClass
  confidence: number
  isUncertain: boolean
  size?: 'sm' | 'md'
}

export function PredictionBadge({
  tumorClass,
  confidence,
  isUncertain,
  size = 'md',
}: PredictionBadgeProps) {
  const color = TUMOR_CLASS_COLORS[tumorClass]
  const label = TUMOR_CLASS_DISPLAY[tumorClass]
  const padding = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'

  return (
    <span
      className={`badge ${padding} font-semibold`}
      style={{ backgroundColor: `${color}15`, color }}
    >
      {label}
    </span>
  )
}

export function ConfidenceBadge({
  confidence,
  isUncertain,
}: {
  confidence: number
  isUncertain: boolean
}) {
  const { text, color, bgColor } = getUncertaintyLabel(confidence)
  const Icon = confidence >= 0.9 ? CheckCircle : confidence >= 0.75 ? AlertCircle : AlertTriangle

  return (
    <span className={`badge ${color} ${bgColor}`}>
      <Icon className="h-3.5 w-3.5" />
      {text} ({formatConfidence(confidence)})
    </span>
  )
}
