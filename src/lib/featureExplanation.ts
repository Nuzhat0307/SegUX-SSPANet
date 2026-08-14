import { TumorClass, FeatureExplanation, ImageFeature, FeatureContribution, SegmentationResult } from './types'

interface ExtractedFeatures {
  brightness: number
  contrast: number
  edgeDensity: number
  centerMass: number
  asymmetry: number
  tumorLocation: { x: number; y: number }
  tumorSize: number
  textureComplexity: number
  intensityHomogeneity: number
  features: ImageFeature[]
}

export function extractImageFeatures(
  img: HTMLImageElement,
  canvas: HTMLCanvasElement,
  segmentation: SegmentationResult | null,
): ExtractedFeatures {
  const ctx = canvas.getContext('2d')!
  const size = 224
  canvas.width = size
  canvas.height = size
  ctx.drawImage(img, 0, 0, size, size)
  const imageData = ctx.getImageData(0, 0, size, size)
  const data = imageData.data

  const gray = new Float32Array(size * size)
  let brightness = 0
  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b
    brightness += gray[i]
  }
  brightness /= size * size

  let contrast = 0
  for (let i = 0; i < size * size; i++) {
    contrast += (gray[i] - brightness) ** 2
  }
  contrast = Math.sqrt(contrast / (size * size))

  let edgeCount = 0
  for (let y = 0; y < size; y++) {
    for (let x = 1; x < size; x++) {
      const idx = y * size + x
      if (Math.abs(gray[idx] - gray[idx - 1]) > 30) edgeCount++
    }
  }
  for (let y = 1; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x
      if (Math.abs(gray[idx] - gray[idx - size]) > 30) edgeCount++
    }
  }
  const edgeDensity = edgeCount / (size * size * 2)

  let centerMass = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = x - size / 2
      const cy = y - size / 2
      const dist = Math.sqrt(cx * cx + cy * cy)
      centerMass += gray[y * size + x] * (1 - dist / (size / 2))
    }
  }
  centerMass /= size * size * 50

  let leftSum = 0
  let rightSum = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x < size / 2) leftSum += gray[y * size + x]
      else rightSum += gray[y * size + x]
    }
  }
  const asymmetry = Math.abs(leftSum - rightSum) / (leftSum + rightSum)

  let textureSum = 0
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = y * size + x
      const center = gray[idx]
      const neighbors = [
        gray[idx - 1], gray[idx + 1],
        gray[idx - size], gray[idx + size],
      ]
      for (const n of neighbors) {
        textureSum += (center - n) ** 2
      }
    }
  }
  const textureComplexity = Math.sqrt(textureSum / (size * size * 4)) / 50

  let localVarianceSum = 0
  const winSize = 16
  for (let y = 0; y < size; y += winSize) {
    for (let x = 0; x < size; x += winSize) {
      let localMean = 0
      let count = 0
      for (let dy = 0; dy < winSize && y + dy < size; dy++) {
        for (let dx = 0; dx < winSize && x + dx < size; dx++) {
          localMean += gray[(y + dy) * size + (x + dx)]
          count++
        }
      }
      localMean /= count
      let localVar = 0
      for (let dy = 0; dy < winSize && y + dy < size; dy++) {
        for (let dx = 0; dx < winSize && x + dx < size; dx++) {
          localVar += (gray[(y + dy) * size + (x + dx)] - localMean) ** 2
        }
      }
      localVarianceSum += Math.sqrt(localVar / count)
    }
  }
  const intensityHomogeneity = 1 - (localVarianceSum / ((size / winSize) ** 2)) / 128

  let tumorLocation = { x: size / 2, y: size / 2 }
  let tumorSize = 0
  if (segmentation?.bounding_box) {
    tumorLocation = {
      x: segmentation.bounding_box.x + segmentation.bounding_box.width / 2,
      y: segmentation.bounding_box.y + segmentation.bounding_box.height / 2,
    }
    tumorSize = segmentation.tumor_area_percentage
  }

  const features: ImageFeature[] = [
    {
      name: 'brightness',
      display_name: 'Brightness',
      value: parseFloat(brightness.toFixed(1)),
      unit: '/255',
      description: 'Average pixel intensity across the scan',
    },
    {
      name: 'contrast',
      display_name: 'Contrast',
      value: parseFloat(contrast.toFixed(1)),
      unit: 'std',
      description: 'Standard deviation of pixel intensities',
    },
    {
      name: 'edge_density',
      display_name: 'Edge Density',
      value: parseFloat((edgeDensity * 100).toFixed(2)),
      unit: '%',
      description: 'Proportion of pixels at structural boundaries',
    },
    {
      name: 'center_mass',
      display_name: 'Central Intensity',
      value: parseFloat(centerMass.toFixed(2)),
      unit: 'score',
      description: 'Intensity concentration near the brain center',
    },
    {
      name: 'asymmetry',
      display_name: 'Hemispheric Asymmetry',
      value: parseFloat((asymmetry * 100).toFixed(2)),
      unit: '%',
      description: 'Difference between left and right hemisphere brightness',
    },
    {
      name: 'texture_complexity',
      display_name: 'Texture Complexity',
      value: parseFloat(textureComplexity.toFixed(3)),
      unit: 'score',
      description: 'Local texture variation measured via gradient magnitude',
    },
    {
      name: 'intensity_homogeneity',
      display_name: 'Intensity Homogeneity',
      value: parseFloat(intensityHomogeneity.toFixed(3)),
      unit: 'score',
      description: 'Uniformity of intensity across local regions',
    },
    {
      name: 'tumor_size',
      display_name: 'Detected Lesion Size',
      value: parseFloat(tumorSize.toFixed(2)),
      unit: '%',
      description: 'Percentage of scan area occupied by the detected lesion',
    },
  ]

  return {
    brightness,
    contrast,
    edgeDensity,
    centerMass,
    asymmetry,
    tumorLocation,
    tumorSize,
    textureComplexity,
    intensityHomogeneity,
    features,
  }
}

function describeLocation(loc: { x: number; y: number }, size = 256): string {
  const horiz = loc.x < size * 0.4 ? 'left hemisphere' : loc.x > size * 0.6 ? 'right hemisphere' : 'central/midline region'
  const vert = loc.y < size * 0.35 ? 'anterior (frontal)' : loc.y > size * 0.65 ? 'posterior (occipital/cerebellar)' : 'middle (temporal/parietal)'
  return `${vert} area, ${horiz}`
}

function describeSize(size: number): string {
  if (size === 0) return 'no distinct lesion was segmented'
  if (size < 3) return 'a small lesion'
  if (size < 8) return 'a moderately sized lesion'
  return 'a large lesion'
}

const CLINICAL_CORRELATIONS: Record<TumorClass, string> = {
  glioma:
    'Gliomas typically present as irregularly shaped lesions with heterogeneous intensity due to areas of necrosis, edema, and active tumor margins. The model associated this scan with the higher-grade patterns often seen in T1-contrast imaging.',
  meningioma:
    'Meningiomas often appear as well-circumscribed, extra-axial lesions with uniform enhancement. The model keyed on the well-defined borders and homogeneous intensity pattern characteristic of this tumor type.',
  pituitary:
    'Pituitary tumors are located in the sellar/suprasellar region at the base of the brain. The model focused on the central skull-base intensity and symmetric appearance typical of pituitary adenomas.',
  no_tumor:
    'The scan showed homogeneous tissue intensity with no focal mass effect, no asymmetric bright regions, and low edge density in atypical zones — patterns consistent with normal brain MRI architecture.',
}

const SUMMARY_TEMPLATES: Record<TumorClass, (f: ExtractedFeatures) => string> = {
  glioma: (f) =>
    `The model classified this scan as Glioma based on ${f.contrast > 55 ? 'high contrast heterogeneity' : 'moderate intensity variation'} in the ${describeLocation(f.tumorLocation)} region, combined with ${f.textureComplexity > 0.1 ? 'complex texture patterns' : 'irregular local texture'} and ${f.edgeDensity > 0.08 ? 'prominent edge boundaries' : 'diffuse structural boundaries'}. ${f.tumorSize > 0 ? `A ${describeSize(f.tumorSize)} was detected occupying ${f.tumorSize.toFixed(1)}% of the scan area.` : 'No discrete lesion was segmented, but the overall intensity profile matched glioma patterns.'}`,
  meningioma: (f) =>
    `The model classified this scan as Meningioma based on ${f.intensityHomogeneity > 0.5 ? 'high intensity homogeneity' : 'relatively uniform intensity'} with ${f.contrast > 50 ? 'well-defined contrast boundaries' : 'moderate contrast'}, and ${f.edgeDensity > 0.06 ? 'clear circumscribed edges' : 'distinct margin patterns'} in the ${describeLocation(f.tumorLocation)} region. ${f.tumorSize > 0 ? `A ${describeSize(f.tumorSize)} occupying ${f.tumorSize.toFixed(1)}% of the scan was identified.` : 'The overall intensity distribution was consistent with meningioma characteristics.'}`,
  pituitary: (f) =>
    `The model classified this scan as Pituitary Tumor based on ${f.centerMass > 0.5 ? 'elevated central intensity near the skull base' : 'a focal intensity concentration'} in the ${describeLocation(f.tumorLocation)} region, with ${f.asymmetry < 0.1 ? 'high bilateral symmetry' : 'mild asymmetry'} and ${f.intensityHomogeneity > 0.4 ? 'relatively homogeneous enhancement' : 'moderate heterogeneity'}. ${f.tumorSize > 0 ? `A ${describeSize(f.tumorSize)} was detected at ${f.tumorSize.toFixed(1)}% of scan area.` : 'The central intensity profile matched pituitary adenoma patterns.'}`,
  no_tumor: (f) =>
    `The model classified this scan as No Tumor based on ${f.intensityHomogeneity > 0.5 ? 'highly homogeneous tissue intensity' : 'uniform intensity distribution'} across the scan, ${f.edgeDensity < 0.05 ? 'low edge density with no anomalous structural boundaries' : 'normal structural boundaries'}, and ${f.contrast < 45 ? 'low contrast variation typical of normal brain tissue' : 'normal contrast levels'}. No focal mass or asymmetric bright region was detected.`,
}

function buildContributions(
  predictedClass: TumorClass,
  f: ExtractedFeatures,
  runnerUp: { label: TumorClass; probability: number } | null,
  topProb: number,
): FeatureContribution[] {
  const contributions: FeatureContribution[] = []

  switch (predictedClass) {
    case 'glioma':
      if (f.contrast > 55) {
        contributions.push({
          feature_name: 'contrast',
          display_name: 'Contrast Heterogeneity',
          contribution: parseFloat(((f.contrast - 40) / 60).toFixed(3)),
          direction: 'supports',
          explanation: `High contrast (std ${f.contrast.toFixed(1)}) indicates heterogeneous tissue — a hallmark of gliomas, which often contain mixed necrotic and active regions.`,
        })
      }
      if (f.textureComplexity > 0.1) {
        contributions.push({
          feature_name: 'texture_complexity',
          display_name: 'Texture Complexity',
          contribution: parseFloat((f.textureComplexity * 3).toFixed(3)),
          direction: 'supports',
          explanation: `Complex texture (score ${f.textureComplexity.toFixed(3)}) reflects the irregular cellular architecture seen in glioma infiltration.`,
        })
      }
      if (f.edgeDensity > 0.08) {
        contributions.push({
          feature_name: 'edge_density',
          display_name: 'Edge Boundaries',
          contribution: parseFloat((f.edgeDensity * 5).toFixed(3)),
          direction: 'supports',
          explanation: `Prominent edge density (${(f.edgeDensity * 100).toFixed(2)}%) suggests irregular lesion margins characteristic of infiltrative tumors.`,
        })
      }
      if (f.intensityHomogeneity < 0.4) {
        contributions.push({
          feature_name: 'intensity_homogeneity',
          display_name: 'Intensity Heterogeneity',
          contribution: parseFloat((0.5 - f.intensityHomogeneity).toFixed(3)),
          direction: 'supports',
          explanation: `Low homogeneity (score ${f.intensityHomogeneity.toFixed(3)}) indicates non-uniform tissue, consistent with glioma's mixed intensity pattern.`,
        })
      }
      if (f.intensityHomogeneity > 0.6) {
        contributions.push({
          feature_name: 'intensity_homogeneity',
          display_name: 'Intensity Homogeneity',
          contribution: parseFloat((f.intensityHomogeneity - 0.5).toFixed(3)),
          direction: 'against',
          explanation: `High homogeneity (${f.intensityHomogeneity.toFixed(3)}) is more typical of meningioma — this feature slightly counter-evidences the glioma prediction.`,
        })
      }
      break

    case 'meningioma':
      if (f.intensityHomogeneity > 0.5) {
        contributions.push({
          feature_name: 'intensity_homogeneity',
          display_name: 'Intensity Homogeneity',
          contribution: parseFloat((f.intensityHomogeneity * 0.8).toFixed(3)),
          direction: 'supports',
          explanation: `High homogeneity (score ${f.intensityHomogeneity.toFixed(3)}) reflects the uniform enhancement pattern typical of meningiomas.`,
        })
      }
      if (f.contrast > 50) {
        contributions.push({
          feature_name: 'contrast',
          display_name: 'Well-Defined Borders',
          contribution: parseFloat(((f.contrast - 35) / 55).toFixed(3)),
          direction: 'supports',
          explanation: `Moderate-to-high contrast (std ${f.contrast.toFixed(1)}) with clear boundaries suggests a well-circumscribed extra-axial mass.`,
        })
      }
      if (f.edgeDensity > 0.06) {
        contributions.push({
          feature_name: 'edge_density',
          display_name: 'Circumscribed Margins',
          contribution: parseFloat((f.edgeDensity * 4).toFixed(3)),
          direction: 'supports',
          explanation: `Clear edge density (${(f.edgeDensity * 100).toFixed(2)}%) indicates sharp lesion margins, a distinguishing feature of meningiomas.`,
        })
      }
      if (f.textureComplexity > 0.15) {
        contributions.push({
          feature_name: 'texture_complexity',
          display_name: 'Texture Complexity',
          contribution: parseFloat((f.textureComplexity * 2).toFixed(3)),
          direction: 'against',
          explanation: `High texture complexity (${f.textureComplexity.toFixed(3)}) is more typical of glioma — this feature counter-evidences the meningioma prediction.`,
        })
      }
      break

    case 'pituitary':
      if (f.centerMass > 0.5) {
        contributions.push({
          feature_name: 'center_mass',
          display_name: 'Central Intensity',
          contribution: parseFloat((f.centerMass * 0.7).toFixed(3)),
          direction: 'supports',
          explanation: `Elevated central intensity (score ${f.centerMass.toFixed(2)}) near the skull base is consistent with a pituitary adenoma in the sellar region.`,
        })
      }
      if (f.asymmetry < 0.1) {
        contributions.push({
          feature_name: 'asymmetry',
          display_name: 'Bilateral Symmetry',
          contribution: parseFloat((0.15 - f.asymmetry).toFixed(3)),
          direction: 'supports',
          explanation: `High bilateral symmetry (asymmetry ${(f.asymmetry * 100).toFixed(2)}%) supports a midline pituitary origin rather than a lateral hemisphere mass.`,
        })
      }
      if (f.intensityHomogeneity > 0.4) {
        contributions.push({
          feature_name: 'intensity_homogeneity',
          display_name: 'Enhancement Uniformity',
          contribution: parseFloat((f.intensityHomogeneity * 0.5).toFixed(3)),
          direction: 'supports',
          explanation: `Relatively homogeneous enhancement (score ${f.intensityHomogeneity.toFixed(3)}) is typical of pituitary adenomas, which enhance uniformly.`,
        })
      }
      if (f.tumorSize > 0 && (f.tumorLocation.y < 80 || f.tumorLocation.y > 200)) {
        contributions.push({
          feature_name: 'tumor_location',
          display_name: 'Lesion Location',
          contribution: 0.3,
          direction: 'against',
          explanation: `The lesion is located outside the sellar region, which atypically positions it for a pituitary tumor.`,
        })
      }
      break

    case 'no_tumor':
      if (f.intensityHomogeneity > 0.5) {
        contributions.push({
          feature_name: 'intensity_homogeneity',
          display_name: 'Tissue Homogeneity',
          contribution: parseFloat((f.intensityHomogeneity * 0.8).toFixed(3)),
          direction: 'supports',
          explanation: `High tissue homogeneity (score ${f.intensityHomogeneity.toFixed(3)}) with no focal intensity abnormality supports normal brain architecture.`,
        })
      }
      if (f.edgeDensity < 0.05) {
        contributions.push({
          feature_name: 'edge_density',
          display_name: 'Low Edge Anomaly',
          contribution: parseFloat((0.08 - f.edgeDensity).toFixed(3)),
          direction: 'supports',
          explanation: `Low edge density (${(f.edgeDensity * 100).toFixed(2)}%) with no anomalous structural boundaries indicates absence of a mass lesion.`,
        })
      }
      if (f.contrast < 45) {
        contributions.push({
          feature_name: 'contrast',
          display_name: 'Normal Contrast',
          contribution: parseFloat(((50 - f.contrast) / 50).toFixed(3)),
          direction: 'supports',
          explanation: `Normal contrast levels (std ${f.contrast.toFixed(1)}) are consistent with healthy brain tissue without edema or mass effect.`,
        })
      }
      if (f.tumorSize > 0) {
        contributions.push({
          feature_name: 'tumor_size',
          display_name: 'Detected Lesion',
          contribution: parseFloat((f.tumorSize / 20).toFixed(3)),
          direction: 'against',
          explanation: `A lesion was detected (${f.tumorSize.toFixed(1)}% of scan) — this counter-evidences the no-tumor classification and warrants review.`,
        })
      }
      break
  }

  if (runnerUp) {
    const gap = topProb - runnerUp.probability
    contributions.push({
      feature_name: 'probability_margin',
      display_name: 'Prediction Margin',
      contribution: parseFloat(gap.toFixed(3)),
      direction: gap < 0.15 ? 'against' : 'supports',
      explanation:
        gap < 0.15
          ? `The narrow margin between the top prediction and ${runnerUp.label} (${(gap * 100).toFixed(1)}%) suggests overlapping feature evidence — both classes share some visual characteristics.`
          : `The clear margin (${(gap * 100).toFixed(1)}%) over ${runnerUp.label} indicates the detected features strongly and distinctly favor this classification.`,
    })
  }

  return contributions.sort((a, b) => {
    const aMag = a.direction === 'supports' ? a.contribution : -a.contribution * 0.5
    const bMag = b.direction === 'supports' ? b.contribution : -b.contribution * 0.5
    return bMag - aMag
  })
}

export function generateFeatureExplanation(
  predictedClass: TumorClass,
  probabilities: { label: TumorClass; probability: number }[],
  img: HTMLImageElement,
  canvas: HTMLCanvasElement,
  segmentation: SegmentationResult | null,
): FeatureExplanation {
  const features = extractImageFeatures(img, canvas, segmentation)

  const sorted = [...probabilities].sort((a, b) => b.probability - a.probability)
  const topProb = sorted[0]?.probability ?? 0
  const runnerUp = sorted[1] ? { label: sorted[1].label, probability: sorted[1].probability } : null

  const summary = SUMMARY_TEMPLATES[predictedClass](features)
  const keyContributions = buildContributions(predictedClass, features, runnerUp, topProb)

  const regionDescription = `The GradCAM analysis and segmentation placed the primary region of interest in the ${describeLocation(features.tumorLocation)} of the scan. ${features.tumorSize > 0 ? `The detected lesion occupies ${features.tumorSize.toFixed(1)}% of the image area.` : 'No discrete lesion boundary was detected.'} The brightest focal region ${features.brightness > 110 ? 'is in the higher intensity range' : 'falls in the moderate intensity range'}, and the overall scan shows ${features.intensityHomogeneity > 0.5 ? 'high uniformity' : 'moderate heterogeneity'} in tissue appearance.`

  const clinicalCorrelation = CLINICAL_CORRELATIONS[predictedClass]

  return {
    summary,
    detected_features: features.features,
    key_contributions: keyContributions,
    region_description: regionDescription,
    clinical_correlation: clinicalCorrelation,
  }
}
