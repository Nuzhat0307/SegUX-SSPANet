"""
PDF Report generator — creates professional diagnostic reports using ReportLab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_RIGHT
from io import BytesIO
import base64
from typing import Any
from loguru import logger


def generate_pdf_report(prediction: Any, patient: Any) -> bytes:
    """Generate a PDF diagnostic report and return bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=20, textColor=HexColor("#1b65f0"),
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=13, textColor=HexColor("#1e293b"),
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7, textColor=HexColor("#64748b"),
    )

    elements = []
    elements.append(Paragraph("SegUX-SSPANet Brain Tumor Diagnosis Report", title_style))
    elements.append(Spacer(1, 5 * mm))
    elements.append(Paragraph("AI-Powered Diagnostic Report", body_style))
    elements.append(Spacer(1, 8 * mm))

    # Patient info
    elements.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ["Name", getattr(patient, "name", "N/A")],
        ["MRN", getattr(patient, "mrn", "N/A")],
        ["Age", f"{getattr(patient, 'age', 'N/A')} years" if getattr(patient, "age", None) else "N/A"],
        ["Scan Date", str(getattr(prediction, "created_at", "N/A"))],
    ]
    t = Table(patient_data, colWidths=[50 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#64748b")),
        ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#1e293b")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8 * mm))

    # Diagnosis
    elements.append(Paragraph("Diagnosis Summary", heading_style))
    pred_class = getattr(prediction, "predicted_class", "unknown")
    pred_display = getattr(prediction, "predicted_class_display", pred_class)
    elements.append(Paragraph(f"<b>Predicted Diagnosis:</b> {pred_display}", body_style))
    elements.append(Spacer(1, 4 * mm))

    # Probabilities
    elements.append(Paragraph("Classification Probabilities", heading_style))
    probs = getattr(prediction, "probabilities", [])
    if isinstance(probs, list):
        prob_data = [["Tumor Type", "Probability"]]
        for p in probs:
            prob_data.append([
                p.get("display_name", p.get("label", "")),
                f"{p.get('probability', 0) * 100:.1f}%",
            ])
        t2 = Table(prob_data, colWidths=[80 * mm, 50 * mm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1b65f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8fafc"), HexColor("#e2e8f0")]),
        ]))
        elements.append(t2)
    elements.append(Spacer(1, 8 * mm))

    # Uncertainty
    elements.append(Paragraph("Uncertainty Estimation", heading_style))
    unc = getattr(prediction, "uncertainty", {})
    if isinstance(unc, dict):
        unc_data = [
            ["Method", unc.get("method", "N/A")],
            ["Confidence", f"{unc.get('confidence', 0) * 100:.1f}%"],
            ["Predictive Entropy", f"{unc.get('predictive_entropy', 0):.4f}"],
            ["Mutual Information", f"{unc.get('mutual_information', 0):.4f}"],
            ["Expert Review", "RECOMMENDED" if unc.get("is_uncertain") else "Not required"],
        ]
        t3 = Table(unc_data, colWidths=[50 * mm, 80 * mm])
        t3.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f1f5f9")),
        ]))
        elements.append(t3)

    elements.append(Spacer(1, 8 * mm))

    # Segmentation
    elements.append(Paragraph("Segmentation Results", heading_style))
    seg = getattr(prediction, "segmentation", {})
    if isinstance(seg, dict):
        seg_data = [
            ["Dice Score", f"{seg.get('dice_score', 0):.4f}"],
            ["Tumor Area", f"{seg.get('tumor_area_percentage', 0):.2f}%"],
        ]
        t4 = Table(seg_data, colWidths=[50 * mm, 80 * mm])
        t4.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9)]))
        elements.append(t4)

    elements.append(Spacer(1, 10 * mm))

    # Disclaimer
    elements.append(Paragraph(
        "This report is generated by the SegUX-SSPANet AI system for research and "
        "educational purposes only. It is NOT a substitute for professional medical "
        "diagnosis. All findings must be validated by a qualified radiologist or neurologist.",
        disclaimer_style,
    ))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
