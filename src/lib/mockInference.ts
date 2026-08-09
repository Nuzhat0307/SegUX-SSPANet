import { PredictionResult } from './types'

const API_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function runInference(
  file: File,
  patientId: string
): Promise<PredictionResult> {
  // Convert uploaded image to base64
  const imageBase64 = await fileToBase64(file)

  const response = await fetch(`${API_URL}/api/v1/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_base64: imageBase64,
      patient_id: patientId,
    }),
  })

  const responseText = await response.text()

  if (!response.ok) {
    throw new Error(
      `AI server error (${response.status}): ${responseText}`
    )
  }

  let data: PredictionResult

  try {
    data = JSON.parse(responseText)
  } catch {
    throw new Error(
      `Invalid response from AI server: ${responseText}`
    )
  }

  return data
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

    reader.onerror = () => {
      reject(new Error('Failed to convert image to base64'))
    }

    reader.readAsDataURL(file)
  })
}