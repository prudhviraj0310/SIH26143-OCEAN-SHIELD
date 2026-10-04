"""
Report Generator: Maritime Screening Case Summary
Generates an analyst-review case summary from automated screening outputs.
"""

import hashlib
import json
import math
import os
from datetime import datetime
from html import escape
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from .falsification import FalsificationAndAbstentionEngine


def _metric(value: Any, specification: str = ".2f", suffix: str = "") -> str:
    number = FalsificationAndAbstentionEngine.finite_number(value)
    return f"{number:{specification}}{suffix}" if number is not None else "Not assessed"


def _text(value: Any) -> str:
    return escape(str(value if value is not None else "Not supplied"))


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _declared_digest(provenance: Dict[str, Any], flat_key: str, nested_key: str) -> Optional[str]:
    nested = provenance.get(nested_key)
    value = provenance.get(flat_key) or (nested.get("sha256") if isinstance(nested, dict) else None)
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
        return value.lower()
    return None


def _digest_preview(value: Optional[str]) -> str:
    return f"<code>{value[:28]}...{value[-8:]}</code>" if value else "NOT_SUPPLIED"


class DossierReportGenerator:
    """
    Builds a maritime pollution investigation case summary (PDF).
    The summary is decision support: it never claims legal authority, a finding of
    liability, or an evidentiary chain of custody for unverified source material.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def compute_merkle_root(leaf_hashes: list) -> str:
        """Computes standard binary Merkle Tree root digest from leaf SHA-256 hashes."""
        if not leaf_hashes:
            return hashlib.sha256(b"").hexdigest()
        current_level = sorted(leaf_hashes)
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]
                next_level.append(hashlib.sha256(combined.encode("utf-8")).hexdigest())
            current_level = next_level
        return current_level[0]

    def generate_pdf_dossier(
        self,
        scenario_data: Dict[str, Any],
        sar_results: Dict[str, Any],
        drift_results: Dict[str, Any],
        ais_results: Dict[str, Any],
        filename: Optional[str] = None,
        evidence_provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Builds the PDF document and writes it to disk.
        Returns the absolute file path of the generated PDF.
        """
        now = datetime.utcnow()
        ref_id = f"OS-SCREEN-{now.strftime('%Y%m%d_%H%M%S_%f')}-{str(scenario_data.get('id', 'ENV'))[:4].upper()}"

        if not filename:
            filename = f"Ocean_Shield_Case_Summary_{ref_id}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom Palette: Deep Navy (#0d2040), Alert Red (#a81c1c), Slate Grey (#4a5568)
        color_navy = colors.HexColor("#0a192f")
        color_navy_light = colors.HexColor("#1e3a8a")
        color_alert = colors.HexColor("#b91c1c")
        color_text = colors.HexColor("#1e293b")
        color_muted = colors.HexColor("#64748b")
        color_border = colors.HexColor("#cbd5e1")
        color_bg_light = colors.HexColor("#f8fafc")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=color_navy,
            alignment=1  # Centered
        )

        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=color_muted,
            alignment=1
        )

        h1_style = ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=color_navy_light,
            spaceAfter=4,
            spaceBefore=8
        )

        body_style = ParagraphStyle(
            "Body_Custom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=color_text
        )

        body_bold = ParagraphStyle(
            "Body_Bold",
            parent=body_style,
            fontName="Helvetica-Bold"
        )

        alert_box_style = ParagraphStyle(
            "Alert_Box",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=color_alert
        )

        story = []

        # 1. Analyst-review header
        header_text = """
        <b>OCEAN-SHIELD &mdash; MARITIME ENVIRONMENTAL ANALYTICS</b><br/>
        <b>SIH26143 PROTOTYPE &bull; ANALYST-REVIEW CASE SUMMARY</b>
        """
        story.append(Paragraph(header_text, title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph("AUTOMATED SCREENING OUTPUT &mdash; NOT A STATUTORY NOTICE OR FINDING OF LIABILITY", subtitle_style))
        story.append(Paragraph("Requires calibrated imagery, source provenance, qualified analyst review, and competent-authority action.", subtitle_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=color_navy, spaceAfter=8))

        # 2. Case Metadata Table
        culprit = _mapping(ais_results.get("primary_review_lead"))
        slick = _mapping(sar_results.get("primary_slick"))
        origin = _mapping(drift_results.get("origin_release_point"))
        gate = ais_results.get("screening_gate") or culprit.get("abstention_verdict") or ais_results.get("bayesian_legal_gate")
        if not isinstance(gate, dict) or type(gate.get("is_abstention")) is not bool:
            gate = {"status": "NOT_ASSESSED", "is_abstention": True, "decision": "INSUFFICIENT_EVIDENCE",
                    "reason": "No valid evidence-state gate was supplied."}
        gate = dict(gate)
        supplied_integrity = FalsificationAndAbstentionEngine.assess_ais_integrity(culprit.get("spoofing_audit"))
        supplied_stress = _mapping(culprit.get("adversarial_stress_test"))
        if (not supplied_integrity["passed"] or gate.get("coverage_status") != "VALIDATED"
                or supplied_stress.get("status") != "ASSESSED" or supplied_stress.get("stress_passed") is not True):
            gate["is_abstention"] = True
            gate["integrity_status"] = supplied_integrity["status"]
            if supplied_integrity["status"] == "UNAVAILABLE" or supplied_stress.get("status") == "UNAVAILABLE":
                gate["status"] = "UNAVAILABLE"
            elif gate.get("status") not in ("UNAVAILABLE", "INVALID_INPUT"):
                if (supplied_integrity["status"] == "NOT_ASSESSED" or gate.get("coverage_status") != "VALIDATED"
                        or supplied_stress.get("status") != "ASSESSED"):
                    gate["status"] = "NOT_ASSESSED"
            if gate.get("decision") == "SCREENING_LEAD":
                gate["decision"] = "INSUFFICIENT_EVIDENCE"
            gate["reason"] = str(gate.get("reason", "")) + " Incomplete or failed supplied coverage/integrity/sensitivity evidence remains on hold."
        held = gate.get("is_abstention") is not False or gate.get("status") != "ASSESSED"
        review_status = "EVIDENTIARY LEAD WITHHELD" if held else "SCREENING LEAD - ANALYST REVIEW REQUIRED"
        mass_text = "Not inferred from this SAR scene"
        screening_score = slick.get("screening_score")
        screening_text = _metric(screening_score, ".1f", "/100 uncalibrated geometry screen")
        assumed_age = origin.get("assumed_slick_age_hours")
        assumed_age_text = _metric(assumed_age, ".1f", " hours (analyst hypothesis)")
        lead_score = culprit.get("lead_priority_score")
        lead_score_text = _metric(lead_score, ".1f", "/100 uncalibrated lead-priority score")

        meta_data = [
            [
                Paragraph("<b>Dossier Ref:</b>", body_style),
                Paragraph(f"<b>{ref_id}</b>", body_bold),
                Paragraph("<b>Date of Issue:</b>", body_style),
                Paragraph(f"{now.strftime('%d %b %Y, %H:%M UTC')}", body_style)
            ],
            [
                Paragraph("<b>Maritime Sector:</b>", body_style),
                Paragraph(f"{scenario_data.get('region', 'Indian EEZ')}", body_style),
                Paragraph("<b>Review status:</b>", body_style),
                Paragraph(f"<font color='#b45309'><b>{review_status}</b></font>", body_style)
            ],
            [
                Paragraph("<b>Screening candidate:</b>", body_style),
                Paragraph(f"<b>{culprit.get('vessel_name', 'N/A')}</b>", body_bold),
                Paragraph("<b>IMO / MMSI:</b>", body_style),
                Paragraph(f"{culprit.get('imo', 'N/A')} / {culprit.get('mmsi', 'N/A')}", body_style)
            ],
            [
                Paragraph("<b>Flag State:</b>", body_style),
                Paragraph(f"{culprit.get('flag_state', 'Unknown')}", body_style),
                Paragraph("<b>Priority ranking:</b>", body_style),
                Paragraph(f"<font color='#b45309'><b>{lead_score_text}</b></font>", body_bold)
            ]
        ]

        meta_table = Table(meta_data, colWidths=[100, 170, 110, 160])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color_bg_light),
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Paragraph(
            f"<b>Evidence state: {_text(gate.get('status', 'NOT_ASSESSED'))} / {_text(gate.get('decision'))}</b><br/>"
            f"{_text(gate.get('reason', 'Evidence has not been assessed.'))}<br/>"
            f"Receiver coverage: {_text(gate.get('coverage_status', 'NOT_ASSESSED'))}; "
            f"AIS integrity: {_text(gate.get('integrity_status', 'NOT_ASSESSED'))}; "
            f"sensitivity checks: {_text(gate.get('stress_status', 'NOT_ASSESSED'))}.", body_style))
        story.append(Spacer(1, 8))

        # 3. Satellite SAR radar-screening observations
        sat_meta = _mapping(scenario_data.get("satellite_metadata"))
        story.append(Paragraph("1. SATELLITE RADAR REMOTE SENSING (SAR) SCREENING", h1_style))

        sar_data = [
            [
                Paragraph("<b>Sensor Platform:</b>", body_style),
                Paragraph(_text(sat_meta.get('mission')), body_style),
                Paragraph("<b>Acquisition Timestamp:</b>", body_style),
                Paragraph(f"{sat_meta.get('acquisition_time_utc', 'N/A')}", body_style)
            ],
            [
                Paragraph("<b>Swath Mode & Pol:</b>", body_style),
                Paragraph(f"{_text(sat_meta.get('sensor_mode'))} / {_text(sat_meta.get('polarization'))}", body_style),
                Paragraph("<b>Ground Resolution:</b>", body_style),
                Paragraph(_metric(sat_meta.get('pixel_spacing_m'), '.2f', ' meters/pixel'), body_style)
            ],
            [
                Paragraph("<b>Dark-feature centroid:</b>", body_style),
                Paragraph(f"{_metric(_mapping(slick.get('centroid')).get('lat'), '.4f')}, {_metric(_mapping(slick.get('centroid')).get('lon'), '.4f')}", body_style),
                Paragraph("<b>Dark-feature area:</b>", body_style),
                Paragraph(f"<b>{_metric(slick.get('area_km2'), '.2f', ' km&sup2;')}</b>", body_bold)
            ],
            [
                Paragraph("<b>Mass / thickness:</b>", body_style),
                Paragraph(f"<b>{mass_text}</b>", body_bold),
                Paragraph("<b>Geometry screen:</b>", body_style),
                Paragraph(f"<b>{screening_text}</b>", body_style)
            ],
            [
                Paragraph("<b>Conditional weathering:</b>", body_style),
                Paragraph(f"<b>{(drift_results.get('weathering_summary') or {}).get('physical_state', 'Not assessed: oil profile and mass are required')}</b>", body_style),
                Paragraph("<b>Evaporative Loss / Mousse:</b>", body_style),
                Paragraph(f"{(drift_results.get('weathering_summary') or {}).get('evaporated_fraction_pct', 'N/A')}% Evaporated / {(drift_results.get('weathering_summary') or {}).get('water_content_mousse_pct', 'N/A')}% Water Uptake", body_style)
            ]
        ]

        sar_table = Table(sar_data, colWidths=[120, 150, 130, 140])
        sar_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sar_table)
        quality = _mapping(sar_results.get("observability"))
        preview = _mapping(sar_results.get("super_resolution_metadata"))
        story.append(Paragraph(
            f"SAR observability: {_text(quality.get('status', 'NOT_ASSESSED'))}. "
            f"Display preview: {_text(preview.get('status', 'NOT_SUPPLIED'))}; "
            "interpolation, display scaling, or generated neural detail is not a verified sensor-resolution or native-radiometry gain.", body_style))
        story.append(Spacer(1, 8))

        # 4. Conditional reverse transport scenario
        story.append(Paragraph("2. CONDITIONAL HYDRODYNAMIC TRANSPORT SCENARIO", h1_style))

        drift_provenance = _mapping(drift_results.get("provenance"))
        hindcast_text = f"""
        Transport state: <b>{_text(drift_results.get('status', 'NOT_ASSESSED'))}</b>.
        Supplied field: {_text(drift_provenance.get('met_ocean_source', 'Not supplied'))}.
        Mode: {_text(drift_provenance.get('mode', 'Not supplied'))}; coverage state:
        {_text(drift_provenance.get('coverage_status', 'VALIDATED' if drift_provenance.get('coverage_validated') is True else 'NOT_ASSESSED'))}.
        The model propagates a supplied <b>age hypothesis</b> through the supplied field.
        A conditional trajectory is not an inferred release time, source location, or measured uncertainty boundary.
        """
        story.append(Paragraph(hindcast_text, body_style))
        story.append(Spacer(1, 4))

        origin_data = [
            [
                Paragraph("<b>Conditional backtracked point:</b>", body_style),
                Paragraph(f"<b>{_metric(origin.get('lat'), '.5f')}, {_metric(origin.get('lon'), '.5f')}</b>", body_bold),
                Paragraph("<b>Assumed slick age:</b>", body_style),
                Paragraph(f"<b>{assumed_age_text}</b>", body_bold)
            ],
            [
                Paragraph("<b>Conditional time offset:</b>", body_style),
                Paragraph(f"{_metric(origin.get('estimated_t0_hours_relative'), '.2f', 'h relative')} (conditional clock)", body_style),
                Paragraph("<b>Total Hydrodynamic Drift:</b>", body_style),
                Paragraph(_metric(drift_results.get('total_drift_distance_km'), '.2f', ' km displacement'), body_style)
            ]
        ]

        kde_info = _mapping(drift_results.get("kde_origin_contours"))
        contours = kde_info.get("contours")
        contours = contours if isinstance(contours, list) else []
        c95 = next((c for c in contours if isinstance(c, dict)
                    and FalsificationAndAbstentionEngine.finite_number(c.get("level")) is not None
                    and abs(float(c["level"]) - 0.95) < 0.05), None)
        if c95:
            origin_data.append([
                Paragraph("<b>95% simulated-particle KDE:</b>", body_style),
                Paragraph(f"<b>{c95.get('approximate_area_km2', 'N/A')} km&sup2;</b> envelope", body_bold),
                Paragraph("<b>KDE Peak Mode (x&#770;, y&#770;):</b>", body_style),
                Paragraph(f"{_metric(kde_info.get('peak_density_lat'), '.4f')}, {_metric(kde_info.get('peak_density_lon'), '.4f')}", body_style)
            ])

        origin_table = Table(origin_data, colWidths=[140, 150, 120, 130])
        origin_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(origin_table)
        story.append(Spacer(1, 8))

        # 5. AIS Maritime Traffic Re-construction & Kinematic Anomaly
        story.append(Paragraph("3. AIS CORRIDOR SCREENING & NAVIGATION CONTEXT", h1_style))

        kinematics = _mapping(culprit.get("kinematics"))
        cpa = _mapping(culprit.get("closest_approach"))

        ais_evidence_text = f"""
        Analysis of <b>{ais_results.get('total_vessels_in_region', 0)} vessels</b> operating in the sector filtered down to
        <b>{ais_results.get('vessels_evaluated_in_corridor', 0)} corridor candidates</b>. Vessel <b>{culprit.get('vessel_name', 'N/A')}</b>
        is the highest-ranked supplied corridor candidate. Evidence state is <b>{review_status}</b>.
        The uncalibrated proximity ranking is not an allegation or responsibility finding:
        """
        if not culprit:
            ais_evidence_text = f"No AIS review candidate was supplied. Evidence state is <b>{review_status}</b>; corridor screening is not assessed."
        story.append(Paragraph(ais_evidence_text, body_style))
        story.append(Spacer(1, 4))

        breakdown = _mapping(culprit.get("score_breakdown"))
        suspect_rows = [
            [
                Paragraph("<b>Evaluation Metric</b>", body_bold),
                Paragraph("<b>Observed Telemetry Value</b>", body_bold),
                Paragraph("<b>Review Context</b>", body_bold),
                Paragraph("<b>Screen Value</b>", body_bold)
            ],
            [
                Paragraph("Closest Point of Approach (CPA)", body_style),
                Paragraph(f"<b>{_metric(cpa.get('distance_nm'), '.2f', ' NM')}</b> ({_metric(cpa.get('distance_km'), '.2f', ' km')})", body_style),
                Paragraph("Modelled proximity to candidate origin", body_style),
                Paragraph(f"<b>{_metric(breakdown.get('proximity_score'), '.1f')}/100</b>", body_bold)
            ],
            [
                Paragraph("Temporal Coincidence", body_style),
                Paragraph(f"&Delta;t = <b>{_metric(cpa.get('time_diff_h'), '.2f', ' hours')}</b> from conditional origin", body_style),
                Paragraph("Modelled temporal alignment", body_style),
                Paragraph(f"<b>{_metric(breakdown.get('temporal_score'), '.1f')}/100</b>", body_bold)
            ],
            [
                Paragraph("Speed Over Ground (SOG)", body_style),
                Paragraph(f"Speed: <b>{_metric(kinematics.get('cruise_speed'), '.1f')} &rarr; {_metric(kinematics.get('min_speed_near_origin'), '.1f')} kts</b>", body_style),
                Paragraph(f"{kinematics.get('speed_comment', 'Speed deceleration')}", body_style),
                Paragraph(f"<b>{_metric(breakdown.get('speed_anomaly_score'), '.1f')}/100</b>", body_bold)
            ],
            [
                Paragraph("Vessel class", body_style),
                Paragraph(f"{culprit.get('vessel_type', 'Unknown')} (DWT: {culprit.get('dwt_tonnes') or 'not supplied'})", body_style),
                Paragraph("Descriptive metadata only; excluded from lead score", body_style),
                Paragraph("<b>Not scored</b>", body_bold)
            ]
        ]

        # TOPSIS MCDA Multi-Criteria Metric
        if "topsis_closeness_score" in culprit:
            suspect_rows.append([
                Paragraph("TOPSIS Closeness (C<sub>i</sub>)", body_style),
                Paragraph(f"<b>{_metric(culprit.get('topsis_closeness_score'), '.1f')}/100</b> (Rank #{culprit.get('topsis_rank', 1)})", body_style),
                Paragraph("Descriptive MCDA comparison; excluded from proximity lead score", body_style),
                Paragraph(f"<b>{culprit.get('borda_points', '--')} Borda pts</b>", body_bold)
            ])

        # Forensic AIS Spoofing & Integrity Audit
        integrity = FalsificationAndAbstentionEngine.assess_ais_integrity(culprit.get("spoofing_audit"))
        spoof = _mapping(culprit.get("spoofing_audit"))
        suspect_rows.append([
            Paragraph("AIS Integrity / Continuity", body_style),
            Paragraph(f"<b>{_text(integrity['status'])}</b>", body_style),
            Paragraph(_text(integrity["reason"]), body_style),
            Paragraph(f"{_text(spoof.get('valid_position_fixes'))} valid fixes", body_bold)
        ])

        # Entropy/weights are uncalibrated screening metrics, not confidence.
        entropy = _mapping(gate.get("entropy_metrics")).get("normalized_entropy")
        dec_color = "#d97706" if held else "#16a34a"
        suspect_rows.append([
            Paragraph("Screening Evidence Gate", body_style),
            Paragraph(f"<font color='{dec_color}'><b>{_text(gate.get('decision'))}</b></font>", body_style),
            Paragraph(f"Full-hypothesis H<sub>norm</sub>={_metric(entropy, '.6f')} "
                      "(project eligibility &le;0.82). Includes unknown source; weights are uncalibrated.", body_style),
            Paragraph(f"Unknown weight: {_metric(gate.get('unknown_source_weight'), '.6f')}", body_bold)
        ])

        # Adversarial Falsification Stress Test
        adv = _mapping(culprit.get("adversarial_stress_test"))
        if adv:
            adv_color = "#16a34a" if adv.get("stress_passed") is True else "#dc2626"
            suspect_rows.append([
                Paragraph("Adversarial Stress Test", body_style),
                Paragraph(f"<font color='{adv_color}'><b>{_text(adv.get('status', 'NOT_ASSESSED'))}: {_text(adv.get('verdict'))}</b></font>", body_style),
                Paragraph("Project scalar perturbations and AIS audit; not validated error bounds", body_style),
                Paragraph(f"<b>{_text(adv.get('challenges_passed'))}/4 checks passed</b>", body_bold)
            ])

        suspect_table = Table(suspect_rows, colWidths=[140, 120, 190, 90])
        suspect_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(suspect_table)
        story.append(Spacer(1, 8))

        # Stage 4 Forward Counterfactual Verification Forensics
        cf = _mapping(ais_results.get("counterfactual_verification") or drift_results.get("counterfactual_verification"))
        if cf:
            story.append(Paragraph("4. STAGE 4 FORWARD COUNTERFACTUAL HYDRODYNAMIC VERIFICATION", h1_style))
            cf_metrics = cf.get("verification_metrics") or {}
            v_color = "#059669" if cf.get("verdict") == "CONDITIONAL_SPATIAL_AGREEMENT" else ("#d97706" if cf.get("verdict") == "CONDITIONAL_NEARBY_CORRIDOR" else "#dc2626")
            cf_score = _metric(cf_metrics.get("physical_causality_score"), ".1f") if cf_metrics.get("physical_causality_score") is not None else "NOT ESTIMATED"
            cf_centroid = _metric(cf_metrics.get("centroid_distance_km"), ".2f", " km")
            cf_containment = _metric(cf_metrics.get("predicted_containment_percent"), ".1f", "%")
            cf_jaccard = _metric(cf_metrics.get("jaccard_index"), ".3f", " IoU")
            cf_rows = [
                [
                    Paragraph("<b>Forward Simulation Verdict:</b>", body_style),
                    Paragraph(f"<font color='{v_color}'><b>{cf.get('verdict_badge', cf.get('verdict', 'N/A'))}</b></font>", body_bold),
                    Paragraph("<b>Origin confidence:</b>", body_style),
                    Paragraph(f"<b>{cf_score}</b>", body_bold)
                ],
                [
                    Paragraph("<b>Predicted Centroid Error:</b>", body_style),
                    Paragraph(f"<b>{cf_centroid}</b>", body_style),
                    Paragraph("<b>Particle Containment:</b>", body_style),
                    Paragraph(f"<b>{cf_containment}</b>", body_style)
                ],
                [
                    Paragraph("<b>Spatial Jaccard Index:</b>", body_style),
                    Paragraph(f"<b>{cf_jaccard}</b>", body_style),
                    Paragraph("<b>Trajectory Convergence:</b>", body_style),
                    Paragraph(f"<b>{'Yes (reaches slick)' if cf_metrics.get('trajectory_reaches_slick') else 'No'}</b>", body_style)
                ]
            ]
            cf_table = Table(cf_rows, colWidths=[140, 150, 120, 130])
            cf_table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.5, color_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4") if cf.get("verdict") == "CONDITIONAL_SPATIAL_AGREEMENT" else colors.HexColor("#fffbeb")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(cf_table)
            if cf.get("explanation"):
                story.append(Spacer(1, 3))
                story.append(Paragraph(f"<i>Conditional simulation explanation: {_text(cf['explanation'])}</i>", body_style))
            story.append(Spacer(1, 8))

        # 5. Dark Vessel Detection Forensics
        dark_ships = ais_results.get("dark_vessels_detected", [])
        if dark_ships:
            story.append(Paragraph("5. RADAR / AIS MISMATCH REVIEW CUES", h1_style))
            dark_text = f"""
            CFAR point-target screening extracted <b>{len(dark_ships)} candidate contacts</b> without a nearby AIS
            trajectory in the supplied data. This is a cue for analyst verification, not proof that a transponder was disabled.
            """
            story.append(Paragraph(dark_text, body_style))
            story.append(Spacer(1, 4))

            dark_rows = [
                [
                    Paragraph("<b>Radar Target ID</b>", body_bold),
                    Paragraph("<b>Position (Lat, Lon)</b>", body_bold),
                    Paragraph("<b>RCS (&sigma;<sup>0</sup> dB) / Length</b>", body_bold),
                    Paragraph("<b>Coverage Review Cue</b>", body_bold)
                ]
            ]
            for dtgt in dark_ships[:3]:
                dark_rows.append([
                    Paragraph(f"<b>{dtgt.get('target_id')}</b>", body_style),
                    Paragraph(f"{_metric(dtgt.get('lat'), '.4f')}, {_metric(dtgt.get('lon'), '.4f')}", body_style),
                    Paragraph(f"{dtgt.get('estimated_rcs_db')} dB / ~{dtgt.get('estimated_length_m')}m", body_style),
                    Paragraph(f"<font color='#dc2626'><b>{dtgt.get('threat_classification')}</b></font>", body_style)
                ])

            dark_table = Table(dark_rows, colWidths=[110, 150, 130, 150])
            dark_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ef4444")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(dark_table)
            story.append(Spacer(1, 8))

        # 6. Coastal Hazard Warning
        coast_warning = _mapping(drift_results.get("forecast_warning"))
        if coast_warning.get("will_beach"):
            story.append(Paragraph("6. COASTAL HAZARD & BEACHING INTERCEPTION ALERT", h1_style))
            warning_text = f"""
            <b>MODELLED SHORELINE ALERT:</b> Hydrodynamic forecasting estimates possible slick impact along the coastline
            within <b>{coast_warning.get('estimated_time_to_beach_hours', 'N/A')} hours</b>. Interception coordinates:
            {_metric((coast_warning.get('beaching_location') or {}).get('lat'), '.4f')}, {_metric((coast_warning.get('beaching_location') or {}).get('lon'), '.4f')}.
            This scenario output should be reviewed by the responsible incident commander before any response decision.
            """
            warning_box = Table([[Paragraph(warning_text, alert_box_style)]], colWidths=[540])
            warning_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ("BOX", (0, 0), (-1, -1), 1.0, color_alert),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(warning_box)
            story.append(Spacer(1, 8))

        # 7. Screening cannot supply statutory measurements or legal conclusions.
        story.append(Paragraph("7. SOURCE MEASUREMENTS &amp; EXTERNAL LEGAL REVIEW", h1_style))
        marpol_rows = [
            [
                Paragraph("<b>Review Item</b>", body_bold),
                Paragraph("<b>Assessment State</b>", body_bold),
                Paragraph("<b>Required Independent Support</b>", body_bold),
                Paragraph("<b>Automated Finding</b>", body_bold)
            ],
            [
                Paragraph("Oil volume / concentration", body_style),
                Paragraph("<b>NOT_ASSESSED</b>", body_style),
                Paragraph("Validated thickness, sampling and monitoring measurements", body_style),
                Paragraph("<b>Not determined</b>", body_style)
            ],
            [
                Paragraph("Discharge rate / mechanism", body_style),
                Paragraph("<b>NOT_ASSESSED</b>", body_bold),
                Paragraph("Original discharge monitors, equipment and voyage records", body_style),
                Paragraph("<b>Not determined</b>", body_bold)
            ],
            [
                Paragraph("MARPOL applicability / liability", body_style),
                Paragraph("<b>LEGAL_REVIEW_REQUIRED</b>", body_style),
                Paragraph("Vessel, jurisdiction, operational context and competent-authority interpretation", body_style),
                Paragraph("<b>Not determined</b>", body_style)
            ],
            [
                Paragraph("Source custody / authentication", body_style),
                Paragraph("<b>NOT_ASSESSED</b>", body_style),
                Paragraph("Original source bytes, collection records and qualified custodian", body_style),
                Paragraph("<b>External review</b>", body_bold)
            ]
        ]
        marpol_table = Table(marpol_rows, colWidths=[145, 125, 160, 110])
        marpol_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d97706")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(marpol_table)
        story.append(Spacer(1, 8))

        # 8. Recommended analyst next steps
        story.append(Paragraph("8. ANALYST NEXT STEPS", h1_style))
        next_steps = """
        1. Resolve all evidence holds and verify source scene calibration, geolocation and acquisition time.<br/>
        2. Verify receiver coverage, vessel identity, original AIS records and navigation context.<br/>
        3. Seek independent observations and qualified laboratory source-fingerprinting where appropriate.<br/>
        4. Refer corroborated material and original collection records to the competent authority for review.
        """
        story.append(Paragraph(next_steps, body_style))
        story.append(Spacer(1, 8))

        # Hash only declared assets and generated summary bytes. Never substitute
        # synthetic constants or hashes of source labels for missing raw assets.
        story.append(Paragraph("9. DECLARED ASSET DIGESTS &amp; SUMMARY INTEGRITY", h1_style))
        provenance = evidence_provenance if isinstance(evidence_provenance, dict) else {}
        weights_sha256 = _declared_digest(provenance, "neural_weights_sha256", "model")
        sar_sha256 = _declared_digest(provenance, "sar_raster_sha256", "sar")
        ais_sha256 = _declared_digest(provenance, "ais_telemetry_sha256", "ais")
        hycom_sha256 = _declared_digest(provenance, "metocean_grid_sha256", "met_ocean")
        sar_path = sar_results.get("image_path")  # Display label only; no caller-supplied path is read.
        hycom_name = drift_provenance.get("met_ocean_source", origin.get("hydrodynamic_data_source", "Not supplied"))

        # 5. Canonical summary manifest JSON
        canonical_manifest = {
            "dossier_reference": ref_id,
            "generation_time_utc": now.isoformat() + "Z",
            "satellite_radar_sensor": sar_results.get("active_engine", "Sentinel-1 C-SAR"),
            "slick_properties": {
                "area_km2": slick.get("area_km2"),
                "perimeter_km": slick.get("perimeter_km"),
                "age_assessment": slick.get("age_assessment"),
            },
            "hydrodynamic_hindcast": {
                "origin_lat": origin.get("lat"),
                "origin_lon": origin.get("lon"),
                "release_time_rel_h": origin.get("estimated_t0_hours_relative"),
                "inference_status": origin.get("inference_status"),
                "data_source": hycom_name,
                "status": drift_results.get("status", "NOT_ASSESSED"),
                "provenance": drift_provenance,
            },
            "screening_candidate": {
                "mmsi": culprit.get("mmsi"),
                "imo": culprit.get("imo"),
                "vessel_name": culprit.get("vessel_name"),
                "composite_score": culprit.get("composite_suspect_score"),
                "tier": culprit.get("attribution_tier"),
                "evidence_state": gate,
                "integrity_state": integrity,
            },
            "asset_hashes": {
                "sar_raster": sar_sha256,
                "ais_telemetry": ais_sha256,
                "metocean_hycom": hycom_sha256,
                "neural_weights": weights_sha256
            }
        }
        def json_safe(value):
            if isinstance(value, dict):
                return {str(key): json_safe(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        manifest_json = json.dumps(json_safe(canonical_manifest), sort_keys=True, separators=(',', ':'), allow_nan=False)
        manifest_sha256 = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

        leaf_hashes = [digest for digest in (sar_sha256, ais_sha256, hycom_sha256, weights_sha256, manifest_sha256) if digest]
        merkle_root_hash = self.compute_merkle_root(leaf_hashes)

        ledger_rows = [
            [
                Paragraph("<b>Declared Asset Component</b>", body_bold),
                Paragraph("<b>Source / System Asset</b>", body_bold),
                Paragraph("<b>SHA-256 Cryptographic Hash Digest</b>", body_bold)
            ],
            [
                Paragraph("SAR Sensor Raster", body_style),
                Paragraph(_text(os.path.basename(sar_path) if isinstance(sar_path, str) else "Not supplied"), body_style),
                Paragraph(_digest_preview(sar_sha256), body_style)
            ],
            [
                Paragraph("AIS Telemetry Broadcast", body_style),
                Paragraph(f"MMSI: {culprit.get('mmsi')} ({culprit.get('vessel_name', 'Lead')})", body_style),
                Paragraph(_digest_preview(ais_sha256), body_style)
            ],
            [
                Paragraph("Hydrodynamic Met-Ocean Grid", body_style),
                Paragraph(_text(hycom_name), body_style),
                Paragraph(_digest_preview(hycom_sha256), body_style)
            ],
            [
                Paragraph("Declared Model Weights", body_style),
                Paragraph("Caller-declared digest; model use/authenticity not verified", body_style),
                Paragraph(_digest_preview(weights_sha256), body_style)
            ],
            [
                Paragraph("Incident Case Manifest JSON", body_style),
                Paragraph("canonical_manifest.json", body_style),
                Paragraph(f"<code>{manifest_sha256[:28]}...{manifest_sha256[-8:]}</code>", body_style)
            ]
        ]
        ledger_table = Table(ledger_rows, colWidths=[140, 150, 250])
        ledger_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ledger_table)
        story.append(Spacer(1, 6))

        # Merkle Root Box
        merkle_text = f"""
        <b>SUMMARY / DECLARED-DIGEST MERKLE ROOT ({len(leaf_hashes)} AVAILABLE LEAVES):</b><br/>
        <code>{merkle_root_hash}</code><br/>
        <i>This digest binds only the generated summary and any valid caller-declared asset digests listed above.
        Missing raw-asset digests remain NOT_SUPPLIED. Source authentication, collection history and custody
        are not established by a hash; no legal certification or liability finding is generated.</i>
        """
        merkle_box = Table([[Paragraph(merkle_text, alert_box_style)]], colWidths=[540])
        merkle_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#16a34a")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(merkle_box)

        # Build document
        doc.build(story)
        return os.path.abspath(filepath)
