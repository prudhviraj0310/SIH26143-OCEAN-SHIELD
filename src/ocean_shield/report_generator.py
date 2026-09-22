"""
Report Generator: Indian Coast Guard & NTRO Formal Marine Pollution Dossier
Generates an analyst-review case summary from automated screening outputs.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class DossierReportGenerator:
    """
    Builds a maritime pollution investigation case summary (PDF).
    The summary is decision support: it never claims legal authority, a finding of
    liability, or an evidentiary chain of custody for unverified source material.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        ref_id = f"ICG-MEP-DOSSIER-{now.strftime('%Y%m%d')}-{scenario_data.get('id', 'ENV')[:4].upper()}"

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
        culprit = ais_results.get("primary_review_lead") or {}
        slick = sar_results.get("primary_slick") or {}
        origin = drift_results.get("origin_release_point") or {}
        mass = slick.get("estimated_mass_tonnes")
        mass_text = f"{mass:.1f} Metric Tonnes" if isinstance(mass, (int, float)) else "Not estimated from this SAR scene"
        screening_score = slick.get("screening_score")
        screening_text = f"{screening_score:.1f}/100 uncalibrated geometry screen" if isinstance(screening_score, (int, float)) else "Not available"
        assumed_age = origin.get("assumed_slick_age_hours")
        assumed_age_text = f"{assumed_age:.1f} hours (analyst hypothesis)" if isinstance(assumed_age, (int, float)) else "Not supplied"
        lead_score = culprit.get("lead_priority_score")
        lead_score_text = f"{lead_score:.1f}/100 lead-priority score" if isinstance(lead_score, (int, float)) else "No eligible AIS lead"

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
                Paragraph("<font color='#b45309'><b>SCREENING LEAD &mdash; REVIEW REQUIRED</b></font>", body_style)
            ],
            [
                Paragraph("<b>Target Vessel:</b>", body_style),
                Paragraph(f"<b>{culprit.get('vessel_name', 'N/A')}</b>", body_bold),
                Paragraph("<b>IMO / MMSI:</b>", body_style),
                Paragraph(f"{culprit.get('imo', 'N/A')} / {culprit.get('mmsi', 'N/A')}", body_style)
            ],
            [
                Paragraph("<b>Flag State:</b>", body_style),
                Paragraph(f"{culprit.get('flag_state', 'Unknown')}", body_style),
                Paragraph("<b>Attribution rank:</b>", body_style),
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
        story.append(Spacer(1, 8))

        # 3. Satellite SAR radar-screening observations
        sat_meta = scenario_data.get("satellite_metadata", {})
        story.append(Paragraph("1. SATELLITE RADAR REMOTE SENSING (SAR) SCREENING", h1_style))

        sar_data = [
            [
                Paragraph("<b>Sensor Platform:</b>", body_style),
                Paragraph(f"{sat_meta.get('mission', 'Sentinel-1 C-SAR')}", body_style),
                Paragraph("<b>Acquisition Timestamp:</b>", body_style),
                Paragraph(f"{sat_meta.get('acquisition_time_utc', 'N/A')}", body_style)
            ],
            [
                Paragraph("<b>Swath Mode & Pol:</b>", body_style),
                Paragraph(f"{sat_meta.get('sensor_mode', 'IW')} / {sat_meta.get('polarization', 'VV')}", body_style),
                Paragraph("<b>Ground Resolution:</b>", body_style),
                Paragraph(f"{sat_meta.get('pixel_spacing_m', 10.0)} meters/pixel", body_style)
            ],
            [
                Paragraph("<b>Observed Slick Centroid:</b>", body_style),
                Paragraph(f"{slick.get('centroid', {}).get('lat', 0):.4f}&deg; N, {slick.get('centroid', {}).get('lon', 0):.4f}&deg; E", body_style),
                Paragraph("<b>Total Spill Surface Area:</b>", body_style),
                Paragraph(f"<b>{slick.get('area_km2', 0):.2f} km&sup2;</b>", body_bold)
            ],
            [
                Paragraph("<b>Mass / thickness:</b>", body_style),
                Paragraph(f"<b>{mass_text}</b>", body_bold),
                Paragraph("<b>Geometry screen:</b>", body_style),
                Paragraph(f"<b>{screening_text}</b>", body_style)
            ],
            [
                Paragraph("<b>ADIOS Weathering State:</b>", body_style),
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
        story.append(Spacer(1, 8))

        # 4. Conditional reverse transport scenario
        story.append(Paragraph("2. CONDITIONAL HYDRODYNAMIC TRANSPORT SCENARIO", h1_style))

        drift_provenance = drift_results.get("provenance", {})
        hindcast_text = f"""
        Using an advection-diffusion Lagrangian numerical formulation with the supplied met-ocean field
        (&Delta;u={drift_provenance.get('current_u_ms', scenario_data.get('ocean_conditions', {}).get('base_current_u', 0)):.2f} m/s,
        &Delta;v={drift_provenance.get('current_v_ms', scenario_data.get('ocean_conditions', {}).get('base_current_v', 0)):.2f} m/s) and Ekman surface windage
        (3.2% Stokes drift factor with 15&deg; Coriolis deflection), the slick plume trajectory was hindcasted backwards in time.
        The model propagates the supplied <b>age hypothesis</b> backward through the validated input field. It does not infer a release time or establish a source location; uncertainty must be quantified against authoritative current, wind, and imagery inputs before operational use:
        """
        story.append(Paragraph(hindcast_text, body_style))
        story.append(Spacer(1, 4))

        origin_data = [
            [
                Paragraph("<b>Conditional backtracked point:</b>", body_style),
                Paragraph(f"<b>{origin.get('lat', 0):.5f}&deg; N, {origin.get('lon', 0):.5f}&deg; E</b>", body_bold),
                Paragraph("<b>Assumed slick age:</b>", body_style),
                Paragraph(f"<b>{assumed_age_text}</b>", body_bold)
            ],
            [
                Paragraph("<b>Conditional time offset:</b>", body_style),
                Paragraph(f"T &minus; {abs(origin.get('estimated_t0_hours_relative', 0)):.1f}h (not an inferred discharge time)", body_style),
                Paragraph("<b>Total Hydrodynamic Drift:</b>", body_style),
                Paragraph(f"{drift_results.get('total_drift_distance_km', 0):.2f} km displacement", body_style)
            ]
        ]
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

        kinematics = culprit.get("kinematics", {})
        cpa = culprit.get("closest_approach", {})

        ais_evidence_text = f"""
        Analysis of <b>{ais_results.get('total_vessels_in_region', 0)} vessels</b> operating in the sector filtered down to
        <b>{ais_results.get('vessels_evaluated_in_corridor', 0)} corridor candidates</b>. Vessel <b>{culprit.get('vessel_name', 'N/A')}</b>
        is the highest-priority review lead based solely on spatio-temporal proximity. This is not an allegation or responsibility finding:
        """
        story.append(Paragraph(ais_evidence_text, body_style))
        story.append(Spacer(1, 4))

        breakdown = culprit.get("score_breakdown", {})
        suspect_rows = [
            [
                Paragraph("<b>Evaluation Metric</b>", body_bold),
                Paragraph("<b>Observed Telemetry Value</b>", body_bold),
                Paragraph("<b>Review Context</b>", body_bold),
                Paragraph("<b>Screen Value</b>", body_bold)
            ],
            [
                Paragraph("Closest Point of Approach (CPA)", body_style),
                Paragraph(f"<b>{cpa.get('distance_nm', 0):.2f} NM</b> ({cpa.get('distance_km', 0):.2f} km)", body_style),
                Paragraph("Modelled proximity to candidate origin", body_style),
                Paragraph(f"<b>{breakdown.get('proximity_score', 0)}/100</b>", body_bold)
            ],
            [
                Paragraph("Temporal Coincidence", body_style),
                Paragraph(f"&Delta;t = <b>{cpa.get('time_diff_h', 0):.2f} hours</b> from origin", body_style),
                Paragraph("Modelled temporal alignment", body_style),
                Paragraph(f"<b>{breakdown.get('temporal_score', 0)}/100</b>", body_bold)
            ],
            [
                Paragraph("Speed Over Ground (SOG)", body_style),
                Paragraph(f"Drop: <b>{kinematics.get('cruise_speed', 0):.1f} &rarr; {kinematics.get('min_speed_near_origin', 0):.1f} kts</b>", body_style),
                Paragraph(f"{kinematics.get('speed_comment', 'Speed deceleration')}", body_style),
                Paragraph(f"<b>{breakdown.get('speed_anomaly_score', 0)}/100</b>", body_bold)
            ],
            [
                Paragraph("Vessel class", body_style),
                Paragraph(f"{culprit.get('vessel_type', 'Unknown')} (DWT: {culprit.get('dwt_tonnes') or 'not supplied'})", body_style),
                Paragraph("Descriptive metadata only; excluded from lead score", body_style),
                Paragraph("<b>Not scored</b>", body_bold)
            ]
        ]

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

        # Dark Vessel Detection Forensics
        dark_ships = ais_results.get("dark_vessels_detected", [])
        if dark_ships:
            story.append(Paragraph("4. NON-COOPERATIVE / DARK VESSEL RADAR SURVEILLANCE AUDIT", h1_style))
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
                    Paragraph("<b>Threat Classification</b>", body_bold)
                ]
            ]
            for dtgt in dark_ships[:3]:
                dark_rows.append([
                    Paragraph(f"<b>{dtgt.get('target_id')}</b>", body_style),
                    Paragraph(f"{dtgt.get('lat'):.4f}&deg; N, {dtgt.get('lon'):.4f}&deg; E", body_style),
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
        coast_warning = drift_results.get("forecast_warning", {}) or {}
        if coast_warning.get("will_beach"):
            story.append(Paragraph("5. COASTAL HAZARD & BEACHING INTERCEPTION ALERT", h1_style))
            warning_text = f"""
            <b>MODELLED SHORELINE ALERT:</b> Hydrodynamic forecasting estimates possible slick impact along the coastline
            within <b>{coast_warning.get('estimated_time_to_beach_hours', 'N/A')} hours</b>. Interception coordinates:
            {coast_warning.get('beaching_location', {}).get('lat', 0):.4f}&deg; N, {coast_warning.get('beaching_location', {}).get('lon', 0):.4f}&deg; E.
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

        # 7. Recommended analyst next steps
        story.append(Paragraph("5. ANALYST NEXT STEPS", h1_style))
        next_steps = f"""
        1. Obtain original calibrated SAR/EO products and preserve their source metadata and hashes.<br/>
        2. Re-run drift with authoritative, time-aligned current and wind fields and report uncertainty bounds.<br/>
        3. Verify AIS completeness, vessel identity, and timing against the original provider export; obtain independent corroboration before contacting
        vessel <b>{culprit.get('vessel_name', 'N/A')} (IMO: {culprit.get('imo', 'N/A')})</b>.<br/>
        4. Refer any enforcement decision to the competent authority and applicable law.
        """
        story.append(Paragraph(next_steps, body_style))
        story.append(Spacer(1, 10))

        # 8. Reproducibility fingerprint. This is deliberately not a legal
        # certificate, signature, or assertion of source-system integrity.
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
                "data_source": origin.get("hydrodynamic_data_source", "HYCOM NetCDF")
            },
            "attributed_vessel": {
                "mmsi": culprit.get("mmsi"),
                "imo": culprit.get("imo"),
                "vessel_name": culprit.get("vessel_name"),
                "composite_score": culprit.get("composite_suspect_score"),
                "tier": culprit.get("attribution_tier")
            }
        }
        manifest_json = json.dumps(canonical_manifest, sort_keys=True, separators=(',', ':'))
        manifest_sha256 = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        
        fingerprint_body = ParagraphStyle(
            "FingerprintBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=color_text
        )

        provenance_note = "No source provenance was supplied."
        if evidence_provenance:
            provenance_note = "Source inputs recorded: " + ", ".join(sorted(evidence_provenance.keys())) + "."
        fingerprint_text = f"""
        <b>REPRODUCIBILITY FINGERPRINT &mdash; NOT A LEGAL CERTIFICATE:</b><br/>
        This SHA-256 value fingerprints the displayed screening summary at generation time. It does not authenticate source imagery,
        establish chain of custody, or substitute for a qualified certificate or authority review.<br/>
        <b>Summary manifest SHA-256:</b> <code>{manifest_sha256}</code><br/>
        {provenance_note}
        """

        footer_table = Table([[Paragraph(fingerprint_text, fingerprint_body)]], colWidths=[540])
        footer_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, color_navy_light),
            ("BACKGROUND", (0, 0), (-1, -1), color_bg_light),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(footer_table)

        # Build document
        doc.build(story)
        return os.path.abspath(filepath)
