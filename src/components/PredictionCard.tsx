import { PredictionResult, TUMOR_CLASS_COLORS, TUMOR_CLASS_DISPLAY } from '../lib/types'
import { PredictionBadge, ConfidenceBadge } from './PredictionBadge'
import { formatDate, formatProbability } from '../lib/utils'
import { Link } from 'react-router-dom'
import { ChevronRight, Clock } from 'lucide-react'

interface PredictionCardProps {
  prediction: PredictionResult
  patientName?: string
}

export default function PredictionCard({ prediction, patientName }: PredictionCardProps) {
  return (
    <Link
      to={`/prediction/${prediction.id}`}
      className="card group block overflow-hidden transition-all hover:shadow-md hover:border-primary-200"
    >
      <div className="flex gap-4 p-4">
        <div className="relative h-24 w-24 flex-shrink-0 overflow-hidden rounded-lg bg-neutral-900">
          <img
            src={prediction.image_base64}
            alt="MRI scan"
            className="h-full w-full object-cover"
          />
          <div
            className="absolute bottom-1 left-1 right-1 rounded px-1.5 py-0.5 text-center text-[10px] font-bold text-white"
            style={{ backgroundColor: TUMOR_CLASS_COLORS[prediction.predicted_class] }}
          >
            {TUMOR_CLASS_DISPLAY[prediction.predicted_class]}
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col justify-between">
          <div>
            <div className="flex items-start justify-between gap-2">
              <div>
                {patientName && (
                  <p className="truncate text-sm font-semibold text-neutral-900">{patientName}</p>
                )}
                <p className="text-xs text-neutral-500">
                  {formatDate(prediction.created_at)}
                </p>
              </div>
              <ChevronRight className="h-5 w-5 flex-shrink-0 text-neutral-300 transition-transform group-hover:translate-x-0.5 group-hover:text-primary-500" />
            </div>

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <PredictionBadge
                tumorClass={prediction.predicted_class}
                confidence={prediction.uncertainty.confidence}
                isUncertain={prediction.uncertainty.is_uncertain}
                size="sm"
              />
              <ConfidenceBadge
                confidence={prediction.uncertainty.confidence}
                isUncertain={prediction.uncertainty.is_uncertain}
              />
            </div>

            <div className="mt-2 flex items-center gap-3 text-xs text-neutral-400">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {prediction.inference_time_ms}ms
              </span>
              <span>{prediction.model_version}</span>
            </div>
          </div>
        </div>
      </div>
    </Link>
  )
}
