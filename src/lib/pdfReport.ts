import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'
import {
  PredictionResult,
  Patient,
  TUMOR_CLASS_DISPLAY,
  TUMOR_CLASS_DESCRIPTIONS,
} from './types'

function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function safeFixed(value: unknown, digits = 2, fallback = 'N/A'): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : fallback
}

function probabilityText(value: unknown): string {
  const n = safeNumber(value, NaN)
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : 'N/A'
}

function confidenceText(value: unknown): string {
  const n = safeNumber(value, NaN)
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : 'N/A'
}

function dateText(value: unknown): string {
  if (!value) return 'Not specified'
  const date = new Date(String(value))
  if (Number.isNaN(date.getTime())) return 'Not specified'
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
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

  const addSectionTitle = (title: string, y: number): number => {
    if (y > 260) {
      doc.addPage()
      y = margin
    }
    doc.setTextColor(30, 41, 59)
    doc.setFontSize(13)
    doc.setFont('helvetica', 'bold')
    doc.text(title, margin, y)
    y += 2
    doc.setDrawColor(226, 232, 240)
    doc.line(margin, y, pageWidth - margin, y)
    return y + 7
  }

  // Header
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

  doc.setFontSize(8)
  doc.text(`Report ID: ${String(prediction.id || '').slice(0, 8).toUpperCase() || 'N/A'}`, pageWidth - margin, 12, { align: 'right' })
  doc.text(`Date: ${dateText(prediction.created_at)}`, pageWidth - margin, 18, { align: 'right' })
  doc.text(`Model: ${prediction.model_version || 'SegUX-SSPANet-v1.0.0'}`, pageWidth - margin, 24, { align: 'right' })

  let y = 45

  // Patient Information
  y = addSectionTitle('Patient Information', y)
  autoTable(doc, {
    startY: y,
    theme: 'plain',
    styles: { fontSize: 9, cellPadding: 2 },
    body: [
      ['Patient Name', patient?.name || 'Not specified'],
      ['Medical Record Number', patient?.mrn || 'Not specified'],
      ['Age', patient?.age != null ? `${patient.age} years` : 'Not specified'],
      ['Gender', patient?.gender ? String(patient.gender).charAt(0).toUpperCase() + String(patient.gender).slice(1) : 'Not specified'],
      ['Scan Date', dateText(prediction.created_at)],
    ],
  })
  y = (doc as any).lastAutoTable.finalY + 10

  // Diagnosis Summary
  y = addSectionTitle('Diagnosis Summary', y)
  const tumorClass = prediction.predicted_class
  const tumorColor: [number, number, number] =
    tumorClass === 'glioma'
      ? [220, 38, 38]
      : tumorClass === 'meningioma'
        ? [249, 115, 22]
        : tumorClass === 'pituitary'
          ? [139, 92, 246]
          : [34, 197, 94]

  const diagnosisName = TUMOR_CLASS_DISPLAY[tumorClass] || prediction.predicted_class_display || tumorClass || 'Unknown'
  const diagnosisDescription = TUMOR_CLASS_DESCRIPTIONS[tumorClass] || 'No diagnostic description is available for this prediction.'

  doc.setFillColor(...tumorColor)
  doc.roundedRect(margin, y, contentWidth, 12, 2, 2, 'F')
  doc.setTextColor(255, 255, 255)
  doc.setFontSize(11)
  doc.setFont('helvetica', 'bold')
  doc.text(`Predicted Diagnosis: ${diagnosisName}`, margin + 4, y + 8)
  y += 18

  doc.setTextColor(51, 65, 85)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  const descLines = doc.splitTextToSize(diagnosisDescription, contentWidth)
  doc.text(descLines, margin, y)
  y += descLines.length * 5 + 6

  // Classification probabilities
  y = addSectionTitle('Classification Probabilities', y)
  const probabilities = Array.isArray(prediction.probabilities) ? prediction.probabilities : []
  autoTable(doc, {
    startY: y,
    head: [['Tumor Type', 'Probability', 'Rank']],
    body: probabilities.length
      ? probabilities.map((p, i) => [
          p.displayName || TUMOR_CLASS_DISPLAY[p.label] || p.label || 'Unknown',
          probabilityText(p.probability),
          i === 0 ? 'Primary prediction' : String(i + 1),
        ])
      : [['No probability data available', 'N/A', 'N/A']],
    theme: 'striped',
    headStyles: { fillColor: [27, 101, 240], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
  })
  y = (doc as any).lastAutoTable.finalY + 10

  // Uncertainty
  y = addSectionTitle('Uncertainty Estimation (Monte Carlo Dropout)', y)
  const uncertainty = prediction.uncertainty
  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: { fillColor: [13, 148, 136], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
    body: [
      ['Method', uncertainty?.method ? String(uncertainty.method).replace(/_/g, ' ') : 'Monte Carlo Dropout'],
      ['MC Samples', String(uncertainty?.num_samples ?? 'N/A')],
      ['MC-Dropout Confidence', confidenceText(uncertainty?.confidence)],
      ['Predictive Entropy', safeFixed(uncertainty?.predictive_entropy, 4)],
      ['Mutual Information', safeFixed(uncertainty?.mutual_information, 4)],
      ['Expert Review', uncertainty?.is_uncertain ? 'RECOMMENDED' : 'Not required'],
    ],
  })
  y = (doc as any).lastAutoTable.finalY + 10

  // Segmentation
  y = addSectionTitle('Segmentation Results (U-Net)', y)
  const segmentation = prediction.segmentation
  const dice = safeFixed(segmentation?.dice_score, 4)
  const tumorArea = safeFixed(segmentation?.tumor_area_percentage, 2)
  const tumorPixels = segmentation?.tumor_area_pixels
  const bbox = segmentation?.bounding_box
  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: { fillColor: [27, 101, 240], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
    body: [
      ['Dice Score', dice],
      ['Tumor Area (%)', tumorArea === 'N/A' ? 'N/A' : `${tumorArea}%`],
      ['Tumor Area (pixels)', tumorPixels != null ? String(tumorPixels) : 'N/A'],
      ['Bounding Box', bbox ? `X:${bbox.x}, Y:${bbox.y}, W:${bbox.width}, H:${bbox.height}` : 'N/A (no bounding box)'],
    ],
  })
  y = (doc as any).lastAutoTable.finalY + 10

  // Explainability
  y = addSectionTitle('Explainability Analysis', y)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)
  const explainText =
    'This analysis utilizes GradCAM-based methods to identify image regions that influence the model prediction. GradCAM provides coarse localization, GradCAM++ improves localization using weighted gradients, and EigenGradCAM provides a gradient-based principal-component explanation.'
  const explainLines = doc.splitTextToSize(explainText, contentWidth)
  doc.text(explainLines, margin, y)
  y += explainLines.length * 5 + 8

  // Feature-based explainability
  const fe = prediction.feature_explanation
  if (fe) {
    y = addSectionTitle('Feature-Based Explainability', y)
    doc.setFontSize(9)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(51, 65, 85)
    const summary = fe.summary || 'No feature summary is available.'
    const summaryLines = doc.splitTextToSize(summary, contentWidth)
    doc.text(summaryLines, margin, y)
    y += summaryLines.length * 5 + 6

    const detectedFeatures = Array.isArray(fe.detected_features) ? fe.detected_features : []
    if (detectedFeatures.length) {
      autoTable(doc, {
        startY: y,
        theme: 'striped',
        headStyles: { fillColor: [13, 148, 136], fontSize: 8 },
        styles: { fontSize: 8, cellPadding: 2 },
        head: [['Feature', 'Value', 'Description']],
        body: detectedFeatures.map((f) => [
          f.display_name || f.name || 'Feature',
          `${f.value ?? 'N/A'} ${f.unit || ''}`.trim(),
          f.description || '',
        ]),
      })
      y = (doc as any).lastAutoTable.finalY + 8
    }

    const contributions = Array.isArray(fe.key_contributions) ? fe.key_contributions : []
    if (contributions.length) {
      if (y > 230) {
        doc.addPage()
        y = margin
      }
      autoTable(doc, {
        startY: y,
        theme: 'striped',
        headStyles: { fillColor: [27, 101, 240], fontSize: 8 },
        styles: { fontSize: 8, cellPadding: 2 },
        head: [['Feature', 'Direction', 'Contribution', 'Explanation']],
        body: contributions.map((c) => {
          const contribution = safeNumber(c.contribution, NaN)
          return [
            c.display_name || c.feature_name || 'Feature',
            c.direction === 'supports' ? 'Supports' : 'Counter-evidence',
            Number.isFinite(contribution) ? `${c.direction === 'supports' ? '+' : '-'}${(contribution * 100).toFixed(1)}%` : 'N/A',
            c.explanation || '',
          ]
        }),
        columnStyles: { 3: { cellWidth: 80 } },
      })
      y = (doc as any).lastAutoTable.finalY + 8
    }

    if (fe.region_description) {
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(27, 101, 240)
      doc.setFontSize(9)
      doc.text('Region of Interest Analysis', margin, y)
      y += 5
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(51, 65, 85)
      const lines = doc.splitTextToSize(fe.region_description, contentWidth)
      doc.text(lines, margin, y)
      y += lines.length * 5 + 5
    }

    if (fe.clinical_correlation) {
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(13, 148, 136)
      doc.text('Clinical Correlation', margin, y)
      y += 5
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(51, 65, 85)
      const lines = doc.splitTextToSize(fe.clinical_correlation, contentWidth)
      doc.text(lines, margin, y)
      y += lines.length * 5 + 8
    }
  }

  // Visual results
  if (y > 180) {
    doc.addPage()
    y = margin
  }
  y = addSectionTitle('Visual Results', y)

  const imgWidth = (contentWidth - 5) / 2
  const imgHeight = imgWidth * 0.75
  try {
    const originalData = prediction.image_base64
    if (originalData?.startsWith('data:image')) {
      doc.addImage(originalData, 'JPEG', margin, y, imgWidth, imgHeight)
      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      doc.text('Original MRI', margin, y + imgHeight + 5)
    }

    const segData = prediction.segmentation?.overlay_base64
    if (segData?.startsWith('data:image')) {
      doc.addImage(segData, 'PNG', margin + imgWidth + 5, y, imgWidth, imgHeight)
      doc.text('Segmentation Overlay', margin + imgWidth + 5, y + imgHeight + 5)
    }
  } catch (err) {
    console.error('Image embedding error:', err)
  }

  y += imgHeight + 15

  try {
    const gradcamImg = prediction.gradcam_results?.[0]
    if (gradcamImg?.overlay_base64?.startsWith('data:image') && y < pageHeight - 70) {
      doc.addImage(gradcamImg.overlay_base64, 'PNG', margin + imgWidth + 5, y, imgWidth, imgHeight * 0.8)
      doc.setFontSize(8)
      doc.text('GradCAM Overlay', margin + imgWidth + 5, y + imgHeight * 0.8 + 5)
    }
  } catch (err) {
    console.error('GradCAM image error:', err)
  }

  // Footer / disclaimer
  const footerY = pageHeight - 25
  doc.setFillColor(241, 245, 249)
  doc.rect(0, footerY, pageWidth, 25, 'F')
  doc.setTextColor(100, 116, 139)
  doc.setFontSize(7)
  doc.setFont('helvetica', 'italic')
  const disclaimer =
    'This report is generated by the SegUX-SSPANet AI system for research and educational purposes only. It is NOT a substitute for professional medical diagnosis. All findings must be validated by a qualified radiologist or neurologist. ' +
    `Inference time: ${safeNumber(prediction.inference_time_ms)}ms | Model: ${prediction.model_version || 'SegUX-SSPANet-v1.0.0'}`
  const disclaimerLines = doc.splitTextToSize(disclaimer, contentWidth)
  doc.text(disclaimerLines, margin, footerY + 8)

  const safeMrn = String(patient?.mrn || 'patient').replace(/[^a-zA-Z0-9_-]/g, '_')
  const safeId = String(prediction.id || 'report').slice(0, 8)
  doc.save(`brain_tumor_report_${safeMrn}_${safeId}.pdf`)
}
