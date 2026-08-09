import { TumorClass, TUMOR_CLASS_COLORS } from './types'

export function formatProbability(p: number): string {
  return `${(p * 100).toFixed(1)}%`
}

export function formatConfidence(c: number): string {
  return `${(c * 100).toFixed(1)}%`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function getTumorColor(cls: TumorClass): string {
  return TUMOR_CLASS_COLORS[cls]
}

export function getUncertaintyLabel(confidence: number): {
  text: string
  color: string
  bgColor: string
} {
  if (confidence >= 0.9) {
    return { text: 'High Confidence', color: 'text-success-700', bgColor: 'bg-success-100' }
  } else if (confidence >= 0.75) {
    return { text: 'Moderate Confidence', color: 'text-warning-700', bgColor: 'bg-warning-100' }
  } else {
    return { text: 'Low Confidence - Review Needed', color: 'text-error-700', bgColor: 'bg-error-100' }
  }
}
