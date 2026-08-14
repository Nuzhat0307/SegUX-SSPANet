import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'
import { PredictionResult, Patient, TUMOR_CLASS_DISPLAY, TUMOR_CLASS_DESCRIPTIONS, FeatureExplanation } from './types'
import { formatProbability, formatConfidence, formatDate } from './utils'

export async function generateReportPDF(prediction: PredictionResult, patient: Patient): Promise<void> {
  const doc = new jsPDF('p', 'mm', 'a4')
  const pageWidth = 210
  const pageHeight = 297
  const margin = 15
  const contentWidth = pageWidth - margin * 2

  // --- Header ---
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

  // Report metadata (right side)
  doc.setFontSize(8)
  doc.text(`Report ID: ${prediction.id.slice(0, 8).toUpperCase()}`, pageWidth - margin, 12, { align: 'right' })
  doc.text(`Date: ${formatDate(prediction.created_at)}`, pageWidth - margin, 18, { align: 'right' })
  doc.text(`Model: ${prediction.model_version}`, pageWidth - margin, 24, { align: 'right' })

  // --- Patient Information ---
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
    styles: { fontSize: 9, cellPadding: 2 },
    body: [
      ['Patient Name', patient.name],
      ['Medical Record Number', patient.mrn],
      ['Age', patient.age ? `${patient.age} years` : 'Not specified'],
      ['Gender', patient.gender ? patient.gender.charAt(0).toUpperCase() + patient.gender.slice(1) : 'Not specified'],
      ['Scan Date', formatDate(prediction.created_at)],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // --- Diagnosis Summary ---
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Diagnosis Summary', margin, y)
  y += 2
  doc.line(margin, y, pageWidth - margin, y)
  y += 8

  // Prediction result box
  const tumorColor: [number, number, number] =
    prediction.predicted_class === 'glioma'
      ? [220, 38, 38]
      : prediction.predicted_class === 'meningioma'
      ? [249, 115, 22]
      : prediction.predicted_class === 'pituitary'
      ? [139, 92, 246]
      : [34, 197, 94]

  doc.setFillColor(tumorColor[0], tumorColor[1], tumorColor[2])
  doc.roundedRect(margin, y, contentWidth, 12, 2, 2, 'F')
  doc.setTextColor(255, 255, 255)
  doc.setFontSize(11)
  doc.setFont('helvetica', 'bold')
  doc.text(
    `Predicted Diagnosis: ${TUMOR_CLASS_DISPLAY[prediction.predicted_class]}`,
    margin + 4,
    y + 8,
  )
  y += 18

  doc.setTextColor(51, 65, 85)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  const descLines = doc.splitTextToSize(
    TUMOR_CLASS_DESCRIPTIONS[prediction.predicted_class],
    contentWidth,
  )
  doc.text(descLines, margin, y)
  y += descLines.length * 5 + 6

  // --- Classification Probabilities ---
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(30, 41, 59)
  doc.text('Classification Probabilities', margin, y)
  y += 2
  doc.setDrawColor(226, 232, 240)
  doc.line(margin, y, pageWidth - margin, y)
  y += 6

  autoTable(doc, {
    startY: y,
    head: [['Tumor Type', 'Probability', 'Confidence Rank']],
    body: prediction.probabilities.map((p, i) => [
      p.displayName,
      formatProbability(p.probability),
      i === 0 ? 'Primary prediction' : `${i + 1}`,
    ]),
    theme: 'striped',
    headStyles: { fillColor: [27, 101, 240], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // --- Uncertainty Estimation ---
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Uncertainty Estimation (Monte Carlo Dropout)', margin, y)
  y += 2
  doc.line(margin, y, pageWidth - margin, y)
  y += 6

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: { fillColor: [13, 148, 136], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
    body: [
      ['Method', prediction.uncertainty.method.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())],
      ['MC Samples', String(prediction.uncertainty.num_samples)],
      ['Confidence', formatConfidence(prediction.uncertainty.confidence)],
      ['Predictive Entropy', prediction.uncertainty.predictive_entropy.toFixed(4)],
      ['Mutual Information', prediction.uncertainty.mutual_information.toFixed(4)],
      ['Expert Review', prediction.uncertainty.is_uncertain ? 'RECOMMENDED' : 'Not required'],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // --- Segmentation Results ---
  if (y > 220) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Segmentation Results (U-Net)', margin, y)
  y += 2
  doc.line(margin, y, pageWidth - margin, y)
  y += 6

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    headStyles: { fillColor: [27, 101, 240], fontSize: 9 },
    styles: { fontSize: 9, cellPadding: 3 },
    body: [
      ['Dice Score', prediction.segmentation.dice_score.toFixed(4)],
      ['Tumor Area (%)', `${prediction.segmentation.tumor_area_percentage.toFixed(2)}%`],
      ['Tumor Area (pixels)', String(prediction.segmentation.tumor_area_pixels)],
      [
        'Bounding Box',
        prediction.segmentation.bounding_box
          ? `X:${prediction.segmentation.bounding_box.x}, Y:${prediction.segmentation.bounding_box.y}, W:${prediction.segmentation.bounding_box.width}, H:${prediction.segmentation.bounding_box.height}`
          : 'N/A (no tumor detected)',
      ],
    ],
  })

  y = (doc as any).lastAutoTable.finalY + 10

  // --- Explainability ---
  if (y > 230) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.text('Explainability Analysis', margin, y)
  y += 2
  doc.line(margin, y, pageWidth - margin, y)
  y += 8

  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)
  const explainText = `This analysis utilized three GradCAM variants to identify the image regions most influential to the model's prediction:

1. GradCAM: Uses gradient information flowing into the last convolutional layer to produce coarse localization maps.
2. GradCAM++: Extends GradCAM with weighted gradients for better object localization and handling of multiple objects.
3. EigenGradCAM: Uses principal component analysis of gradients for more stable and robust explanations.`
  const explainLines = doc.splitTextToSize(explainText, contentWidth)
  doc.text(explainLines, margin, y)
  y += explainLines.length * 5 + 8

  // --- Feature-Based Explainability ---
  if (prediction.feature_explanation) {
    const fe = prediction.feature_explanation

    if (y > 200) {
      doc.addPage()
      y = margin
    }

    doc.setFontSize(13)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(30, 41, 59)
    doc.text('Feature-Based Explainability', margin, y)
    y += 2
    doc.line(margin, y, pageWidth - margin, y)
    y += 7

    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(27, 101, 240)
    doc.text('Decision Summary', margin, y)
    y += 5
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(51, 65, 85)
    const summaryLines = doc.splitTextToSize(fe.summary, contentWidth)
    doc.text(summaryLines, margin, y)
    y += summaryLines.length * 5 + 4

    // Detected features table
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(30, 41, 59)
    doc.setFontSize(10)
    doc.text('Detected Image Features', margin, y)
    y += 4

    autoTable(doc, {
      startY: y,
      theme: 'striped',
      headStyles: { fillColor: [13, 148, 136], fontSize: 8 },
      styles: { fontSize: 8, cellPadding: 2 },
      head: [['Feature', 'Value', 'Description']],
      body: fe.detected_features.map((f) => [
        f.display_name,
        `${f.value} ${f.unit}`,
        f.description,
      ]),
    })

    y = (doc as any).lastAutoTable.finalY + 8

    // Key contributions
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(10)
    doc.text('Key Features Driving the Prediction', margin, y)
    y += 4

    autoTable(doc, {
      startY: y,
      theme: 'striped',
      headStyles: { fillColor: [27, 101, 240], fontSize: 8 },
      styles: { fontSize: 8, cellPadding: 2 },
      head: [['Feature', 'Direction', 'Contribution', 'Explanation']],
      body: fe.key_contributions.map((c) => [
        c.display_name,
        c.direction === 'supports' ? 'Supports' : 'Counter-evidence',
        `${c.direction === 'supports' ? '+' : '-'}${(c.contribution * 100).toFixed(1)}%`,
        c.explanation,
      ]),
      columnStyles: { 3: { cellWidth: 80 } },
    })

    y = (doc as any).lastAutoTable.finalY + 6

    // Region description
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(27, 101, 240)
    doc.setFontSize(9)
    doc.text('Region of Interest Analysis', margin, y)
    y += 5
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(51, 65, 85)
    const regionLines = doc.splitTextToSize(fe.region_description, contentWidth)
    doc.text(regionLines, margin, y)
    y += regionLines.length * 5 + 4

    // Clinical correlation
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(13, 148, 136)
    doc.text('Clinical Correlation', margin, y)
    y += 5
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(51, 65, 85)
    const clinicalLines = doc.splitTextToSize(fe.clinical_correlation, contentWidth)
    doc.text(clinicalLines, margin, y)
    y += clinicalLines.length * 5 + 8
  }

  // --- Images ---
  if (y > 180) {
    doc.addPage()
    y = margin
  }

  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(30, 41, 59)
  doc.text('Visual Results', margin, y)
  y += 2
  doc.line(margin, y, pageWidth - margin, y)
  y += 8

  // Add original MRI and segmentation overlay side by side
  const imgWidth = (contentWidth - 5) / 2
  const imgHeight = imgWidth * 0.75

  try {
    // Original MRI
    const originalData = prediction.image_base64
    if (originalData.startsWith('data:image')) {
      doc.addImage(originalData, 'JPEG', margin, y, imgWidth, imgHeight)
      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      doc.text('Original MRI', margin, y + imgHeight + 5)
    }

    // Segmentation overlay
    const segData = prediction.segmentation.overlay_base64
    if (segData.startsWith('data:image')) {
      doc.addImage(segData, 'PNG', margin + imgWidth + 5, y, imgWidth, imgHeight)
      doc.text('Segmentation Overlay', margin + imgWidth + 5, y + imgHeight + 5)
    }
  } catch (err) {
    console.error('Image embedding error:', err)
  }

  y += imgHeight + 15

  // GradCAM overlays (if space permits)
  if (y < pageHeight - 70) {
    try {
      const gradcamImg = prediction.gradcam_results[0]
      if (gradcamImg?.overlay_base64?.startsWith('data:image')) {
        doc.addImage(gradcamImg.overlay_base64, 'PNG', margin + imgWidth + 5, y, imgWidth, imgHeight * 0.8)
        doc.text('GradCAM Overlay', margin + imgWidth + 5, y + imgHeight * 0.8 + 5)
      }
    } catch (err) {
      console.error('GradCAM image error:', err)
    }
  }

  // --- Disclaimer footer ---
  const footerY = pageHeight - 25
  doc.setFillColor(241, 245, 249)
  doc.rect(0, footerY, pageWidth, 25, 'F')
  doc.setTextColor(100, 116, 139)
  doc.setFontSize(7)
  doc.setFont('helvetica', 'italic')
  const disclaimer =
    'This report is generated by the SegUX-SSPANet AI system for research and educational purposes only. ' +
    'It is NOT a substitute for professional medical diagnosis. All findings must be validated by a qualified radiologist or neurologist. ' +
    `Inference time: ${prediction.inference_time_ms}ms | Model: ${prediction.model_version}`
  const disclaimerLines = doc.splitTextToSize(disclaimer, contentWidth)
  doc.text(disclaimerLines, margin, footerY + 8)

  // Save
  const filename = `brain_tumor_report_${patient.mrn}_${prediction.id.slice(0, 8)}.pdf`
  doc.save(filename)
}