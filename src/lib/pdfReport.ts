import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'
import {
  PredictionResult,
  Patient,
  TUMOR_CLASS_DISPLAY,
  TUMOR_CLASS_DESCRIPTIONS,
} from './types'
import { formatProbability, formatConfidence, formatDate } from './utils'

/**
 * Safely format nullable/undefined numeric values.
 * Prevents PDF generation from failing when the backend returns null.
 */
const formatNumber = (
  value: number | null | undefined,
  decimals = 4,
): string => {
  return value != null && Number.isFinite(Number(value))
    ? Number(value).toFixed(decimals)
    : 'N/A'
}

/**
 * Safely convert any value to a display string.
 */
const safeString = (
  value: unknown,
  fallback = 'N/A',
): string => {
  return value !== null && value !== undefined && value !== ''
    ? String(value)
    : fallback
}

/**
 * Check whether a value is a valid base64 image.
 */
const isValidImageData = (
  value: string | null | undefined,
): value is string => {
  return typeof value === 'string' && value.startsWith('data:image')
}

export async function generateReportPDF(
  prediction: PredictionResult,
  patient: Patient,
): Promise<void> {
  const doc = new jsPDF('p', 'mm', 'a4')

  const pageWidth = 210
  const pageHeight = 297
  const margin = 15
  const contentWidth = pageWidth - margin * 2

  // ---------------------------------------------------------
  // Header
  // ---------------------------------------------------------

  doc.setFillColor(27, 101, 240)
  doc.rect(0, 0, pageWidth, 35, 'F')

  doc.setTextColor(255, 255, 255)
  doc.setFontSize(20)
  doc.setFont('helvetica', 'bold')
  doc.text('SegUX-SSPANet', margin, 15)

  doc.setFontSize(10)
  doc.setFont('helvetica', 'normal')
  doc.text('Brain Tumor Diagnosis System', margin, 22)
  doc.text('AI-Powered Diagnostic Report', margin, 28)

  // Report metadata
  doc.setFontSize(8)

  const predictionId = safeString(prediction.id, 'UNKNOWN')
  const modelVersion = safeString(prediction.model_version, 'Unknown')

  doc.text(
    `Report ID: ${predictionId.slice(0, 8).toUpperCase()}`,
    pageWidth - margin,
    12,
    { align: 'right' },
  )

  doc.text(
    `Date: ${formatDate(prediction.created_at)}`,
    pageWidth - margin,
    18,
    { align: 'right' },
  )

  doc.text(
    `Model: ${modelVersion}`,
    pageWidth - margin,
    24,
    { align: 'right' },
  )

  // ---------------------------------------------------------
  // Patient Information
  // ---------------------------------------------------------

  let y = 45

  doc.setTextColor(30, 41, 59)
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Patient Information', margin, y)

  y += 2

  doc.setDrawColor(226, 232, 240)
  doc.line(margin, y, pageWidth - margin, y)

  y += 6

  autoTable(doc, {
    startY: y,
    theme: 'plain',
    styles: {
      fontSize: 9,
      cellPadding: 2,
    },
    body: [
      ['Patient Name', safeString(patient.name)],
      ['Medical Record Number', safeString(patient.mrn)],
      [
        'Age',
        patient.age != null
          ? `${patient.age} years`
          : 'Not specified',
      ],
      [
        'Gender',
        patient.gender
          ? patient.gender.charAt(0).toUpperCase() +
            patient.gender.slice(1)
          : 'Not specified',
      ],
      ['Scan Date', formatDate(prediction.created_at)],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // ---------------------------------------------------------
  // Diagnosis Summary
  // ---------------------------------------------------------

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Diagnosis Summary', margin, y)

  y += 2

  doc.line(margin, y, pageWidth - margin, y)

  y += 8

  const tumorColor: [number, number, number] =
    prediction.predicted_class === 'glioma'
      ? [220, 38, 38]
      : prediction.predicted_class === 'meningioma'
        ? [249, 115, 22]
        : prediction.predicted_class === 'pituitary'
          ? [139, 92, 246]
          : [34, 197, 94]

  doc.setFillColor(
    tumorColor[0],
    tumorColor[1],
    tumorColor[2],
  )

  doc.roundedRect(
    margin,
    y,
    contentWidth,
    12,
    2,
    2,
    'F',
  )

  doc.setTextColor(255, 255, 255)
  doc.setFontSize(11)
  doc.setFont('helvetica', 'bold')

  const diagnosisName =
    TUMOR_CLASS_DISPLAY[prediction.predicted_class] ??
    safeString(prediction.predicted_class, 'Unknown')

  doc.text(
    `Predicted Diagnosis: ${diagnosisName}`,
    margin + 4,
    y + 8,
  )

  y += 18

  doc.setTextColor(51, 65, 85)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')

  const description =
    TUMOR_CLASS_DESCRIPTIONS[prediction.predicted_class] ??
    'No diagnostic description is available.'

  const descLines = doc.splitTextToSize(
    description,
    contentWidth,
  )

  doc.text(descLines, margin, y)

  y += descLines.length * 5 + 6

  // ---------------------------------------------------------
  // Classification Probabilities
  // ---------------------------------------------------------

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(30, 41, 59)

  doc.text(
    'Classification Probabilities',
    margin,
    y,
  )

  y += 2

  doc.setDrawColor(226, 232, 240)
  doc.line(margin, y, pageWidth - margin, y)

  y += 6

  autoTable(doc, {
    startY: y,
    head: [
      ['Tumor Type', 'Probability', 'Confidence Rank'],
    ],
    body: Array.isArray(prediction.probabilities)
  ? prediction.probabilities.map((p, i) => {
      const tumorNames = [
        'Glioma',
        'Meningioma',
        'Pituitary',
        'No Tumor',
      ]

      const tumorType =
        p.displayName ||
        (p as any).class ||
        (p as any).label ||
        (p as any).name ||
        tumorNames[i] ||
        'Unknown'

      return [
        tumorType,
        p.probability != null
          ? formatProbability(p.probability)
          : 'N/A',
        i === 0
          ? 'Primary prediction'
          : `${i + 1}`,
      ]
    })
  : [['N/A', 'N/A', 'N/A']],
    theme: 'striped',
    headStyles: {
      fillColor: [27, 101, 240],
      fontSize: 9,
    },
    styles: {
      fontSize: 9,
      cellPadding: 3,
    },
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // ---------------------------------------------------------
  // Uncertainty Estimation
  // ---------------------------------------------------------

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')

  doc.text(
    'Uncertainty Estimation (Monte Carlo Dropout)',
    margin,
    y,
  )

  y += 2

  doc.line(margin, y, pageWidth - margin, y)

  y += 6

  const uncertainty = prediction.uncertainty

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: {
      fillColor: [13, 148, 136],
      fontSize: 9,
    },
    styles: {
      fontSize: 9,
      cellPadding: 3,
    },
    body: [
      [
        'Method',
        uncertainty?.method
          ? uncertainty.method
              .replace(/_/g, ' ')
              .replace(/\b\w/g, (c) => c.toUpperCase())
          : 'N/A',
      ],
      [
        'MC Samples',
        uncertainty?.num_samples != null
          ? String(uncertainty.num_samples)
          : 'N/A',
      ],
      [
        'Confidence',
        uncertainty?.confidence != null
          ? formatConfidence(uncertainty.confidence)
          : 'N/A',
      ],
      [
        'Predictive Entropy',
        formatNumber(
          uncertainty?.predictive_entropy,
          4,
        ),
      ],
      [
        'Mutual Information',
        formatNumber(
          uncertainty?.mutual_information,
          4,
        ),
      ],
      [
        'Expert Review',
        uncertainty?.is_uncertain
          ? 'RECOMMENDED'
          : 'Not required',
      ],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // ---------------------------------------------------------
  // Segmentation Results
  // ---------------------------------------------------------

  if (y > 220) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')

  doc.text(
    'Segmentation Results (U-Net)',
    margin,
    y,
  )

  y += 2

  doc.line(margin, y, pageWidth - margin, y)

  y += 6

  const segmentation = prediction.segmentation

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: {
      fillColor: [27, 101, 240],
      fontSize: 9,
    },
    styles: {
      fontSize: 9,
      cellPadding: 3,
    },
    body: [
      [
        'Dice Score',
        formatNumber(
          segmentation?.dice_score,
          4,
        ),
      ],
      [
        'Tumor Area (%)',
        segmentation?.tumor_area_percentage != null
          ? `${formatNumber(
              segmentation.tumor_area_percentage,
              2,
            )}%`
          : 'N/A',
      ],
      [
        'Tumor Area (pixels)',
        segmentation?.tumor_area_pixels != null
          ? String(segmentation.tumor_area_pixels)
          : 'N/A',
      ],
      [
        'Bounding Box',
        segmentation?.bounding_box
          ? `X:${segmentation.bounding_box.x}, ` +
            `Y:${segmentation.bounding_box.y}, ` +
            `W:${segmentation.bounding_box.width}, ` +
            `H:${segmentation.bounding_box.height}`
          : 'N/A (no tumor detected)',
      ],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // ---------------------------------------------------------
  // Explainability
  // ---------------------------------------------------------

  if (y > 230) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')

  doc.text(
    'Explainability Analysis',
    margin,
    y,
  )

  y += 2

  doc.line(margin, y, pageWidth - margin, y)

  y += 8

  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)

  const explainText =
    `This analysis utilized three GradCAM variants to identify ` +
    `the image regions most influential to the model's prediction:\n\n` +
    `1. GradCAM: Uses gradient information flowing into the last ` +
    `convolutional layer to produce coarse localization maps.\n` +
    `2. GradCAM++: Extends GradCAM with weighted gradients for ` +
    `better object localization and handling of multiple objects.\n` +
    `3. EigenGradCAM: Uses principal component analysis of gradients ` +
    `for more stable and robust explanations.`

  const explainLines = doc.splitTextToSize(
    explainText,
    contentWidth,
  )

  doc.text(explainLines, margin, y)

  y += explainLines.length * 5 + 6

  // ---------------------------------------------------------
  // Visual Results
  // ---------------------------------------------------------

  if (y > 180) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(30, 41, 59)

  doc.text(
    'Visual Results',
    margin,
    y,
  )

  y += 2

  doc.line(margin, y, pageWidth - margin, y)

  y += 8

  const imgWidth = (contentWidth - 5) / 2
  const imgHeight = imgWidth * 0.75

  let imageAdded = false

  // ---------------------------------------------------------
  // Original MRI
  // ---------------------------------------------------------

  try {
    const originalData = prediction.image_base64

    if (isValidImageData(originalData)) {
      doc.addImage(
        originalData,
        'JPEG',
        margin,
        y,
        imgWidth,
        imgHeight,
      )

      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')

      doc.text(
        'Original MRI',
        margin,
        y + imgHeight + 5,
      )

      imageAdded = true
    }
  } catch (err) {
    console.error(
      'Original MRI embedding error:',
      err,
    )
  }

  // ---------------------------------------------------------
  // Segmentation Overlay
  // ---------------------------------------------------------

  try {
    const segData =
      prediction.segmentation?.overlay_base64

    if (isValidImageData(segData)) {
      doc.addImage(
        segData,
        'PNG',
        margin + imgWidth + 5,
        y,
        imgWidth,
        imgHeight,
      )

      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')

      doc.text(
        'Segmentation Overlay',
        margin + imgWidth + 5,
        y + imgHeight + 5,
      )

      imageAdded = true
    }
  } catch (err) {
    console.error(
      'Segmentation image embedding error:',
      err,
    )
  }

  // Only move down if an image area was actually used.
  if (imageAdded) {
    y += imgHeight + 15
  } else {
    doc.setFontSize(9)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(100, 116, 139)

    doc.text(
      'Visual image data is not available for this prediction.',
      margin,
      y + 8,
    )

    y += 25
  }

  // ---------------------------------------------------------
  // GradCAM Overlay
  // ---------------------------------------------------------

  if (y < pageHeight - 70) {
    try {
      const gradcamImg =
        Array.isArray(prediction.gradcam_results)
          ? prediction.gradcam_results[0]
          : null

      const gradcamData =
        gradcamImg?.overlay_base64

      if (isValidImageData(gradcamData)) {
        doc.addImage(
          gradcamData,
          'PNG',
          margin + imgWidth + 5,
          y,
          imgWidth,
          imgHeight * 0.8,
        )

        doc.setFontSize(8)
        doc.setFont('helvetica', 'normal')

        doc.text(
          'GradCAM Overlay',
          margin + imgWidth + 5,
          y + imgHeight * 0.8 + 5,
        )
      }
    } catch (err) {
      console.error(
        'GradCAM image error:',
        err,
      )
    }
  }

  // ---------------------------------------------------------
  // Disclaimer Footer
  // ---------------------------------------------------------

  const footerY = pageHeight - 25

  doc.setFillColor(241, 245, 249)
  doc.rect(
    0,
    footerY,
    pageWidth,
    25,
    'F',
  )

  doc.setTextColor(100, 116, 139)
  doc.setFontSize(7)
  doc.setFont('helvetica', 'italic')

  const inferenceTime =
    prediction.inference_time_ms != null
      ? `${prediction.inference_time_ms}ms`
      : 'N/A'

  const disclaimer =
    'This report is generated by the SegUX-SSPANet AI system ' +
    'for research and educational purposes only. ' +
    'It is NOT a substitute for professional medical diagnosis. ' +
    'All findings must be validated by a qualified radiologist ' +
    'or neurologist. ' +
    `Inference time: ${inferenceTime} | ` +
    `Model: ${modelVersion}`

  const disclaimerLines = doc.splitTextToSize(
    disclaimer,
    contentWidth,
  )

  doc.text(
    disclaimerLines,
    margin,
    footerY + 8,
  )

  // ---------------------------------------------------------
  // Save PDF
  // ---------------------------------------------------------

  const mrn = safeString(patient.mrn, 'patient')
  const shortId = predictionId.slice(0, 8)

  const filename =
    `brain_tumor_report_${mrn}_${shortId}.pdf`

  doc.save(filename)
}