import { PredictionResult } from './types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function checkBackendHealth(): Promise<void> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 4000)

  try {
    const response = await fetch(`${API_URL}/api/v1/health`, {
      method: 'GET',
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`AI backend returned HTTP ${response.status}.`)
    }

    const data = await response.json()
    if (!data.model_loaded) {
      throw new Error('AI backend is running, but the trained model is not loaded.')
    }
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('AI backend health check timed out. Please make sure the backend is running.')
    }
    if (error instanceof TypeError) {
      throw new Error('AI backend is unavailable. Start the FastAPI server at http://localhost:8000 and try again.')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

export async function runInference(
  file: File,
  patientId: string
): Promise<PredictionResult> {
  const imageBase64 = await fileToBase64(file)

  let response: Response
  try {
    response = await fetch(`${API_URL}/api/v1/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: imageBase64,
        patient_id: patientId,
      }),
    })
  } catch {
    throw new Error('AI backend is unavailable. Start the FastAPI server at http://localhost:8000 and try again.')
  }

  const responseText = await response.text()

  if (!response.ok) {
    let detail = responseText
    try {
      const parsed = JSON.parse(responseText)
      detail = parsed.detail || responseText
    } catch {
      // Keep the raw response when it is not JSON.
    }
    throw new Error(`AI server error (${response.status}): ${detail}`)
  }

  let raw: any
  try {
    raw = JSON.parse(responseText)
  } catch {
    throw new Error(`Invalid response from AI server: ${responseText}`)
  }

  return normalizePredictionResult(raw)
}

function normalizePredictionResult(raw: any): PredictionResult {
  const probabilities = Array.isArray(raw.probabilities)
    ? raw.probabilities.map((p: any) => ({
        label: p.label,
        displayName: p.displayName ?? p.display_name ?? p.label,
        probability: Number(p.probability ?? 0),
      }))
    : []

  return {
    ...raw,
    probabilities,
    predicted_class_display:
      raw.predicted_class_display ?? raw.predicted_class,
    uncertainty: {
      ...raw.uncertainty,
      num_samples: Number(raw.uncertainty?.num_samples ?? 0),
      predictive_entropy: Number(raw.uncertainty?.predictive_entropy ?? 0),
      mutual_information: Number(raw.uncertainty?.mutual_information ?? 0),
      confidence: Number(raw.uncertainty?.confidence ?? 0),
      is_uncertain: Boolean(raw.uncertainty?.is_uncertain),
    },
  } as PredictionResult
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = () => {
      if (typeof reader.result !== 'string') {
        reject(new Error('Failed to read uploaded image'))
        return
      }
      resolve(reader.result)
    }

    reader.onerror = () => reject(new Error('Failed to convert image to base64'))
    reader.readAsDataURL(file)
  })
}
