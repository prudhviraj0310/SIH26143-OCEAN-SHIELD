# OCEAN-SHIELD: Project Overview & Problem Statement

## 1. Problem Statement Details
* **Hackathon**: Smart India Hackathon 2026 (SIH 2026)
* **Problem Statement ID**: **26143 (SIH26143)**
* **Problem Title**: Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.
* **Organization**: **National Technical Research Organisation (NTRO)**
* **Department**: National Technical Research Organisation (NTRO)
* **Category**: Software
* **Theme**: Disaster Management

---

## 2. Core Problem & Real-World Context
Marine oil spills inflict catastrophic, long-term damage on fragile coastal ecosystems, coral reefs, fisheries, and national critical maritime infrastructure (such as Single Point Mooring terminals and desalination plants). 

Historically, maritime oil spills frequently remain **unattributed**. Rogue commercial vessels (such as oil tankers, chemical carriers, and bulk carriers) deliberately discharge bilge waste, tank washings, or bunker fuel at night or during cloudy weather to avoid heavy maritime environmental penalties under MARPOL Annex I. Because standard optical satellites cannot see through dense clouds or darkness, polluters have operated with impunity.

When a spill is finally detected by coastal authorities or local fishermen, hours or days have already passed. Ocean currents and winds have pushed and weathered the oil slick tens of kilometers away from where the crime actually occurred. By that time, the culprit vessel has transited away, and thousands of innocent ships have traversed the corridor, making forensic identification nearly impossible using manual methods.

---

## 3. Official NTRO Requirements Breakdown
NTRO divided Problem Statement 26143 into three distinct technical challenges:

### (a) Satellite Remote Sensing Detection & Characterization
* **Requirement**: Design an intelligent automated pipeline using remote sensing satellite data, such as Synthetic Aperture Radar (SAR) and Electro-Optical (EO) imagery.
* **Deliverable**: Automatically segment and detect the oil slick, reject biogenic/low-wind lookalikes, calculate geometric metrics (spill surface area in $\text{km}^2$, perimeter, circularity, elongation), and estimate the age of the spill.

### (b) Reverse Hydrodynamic Hindcasting & Forward Forecasting
* **Requirement**: Use oceanographic and meteorological data (currents, wind vectors, sea surface temperature) to trace the slick towards its origin point and release time ($t_0$), and predict the future flow of the slick towards coastlines or sensitive ecological zones.
* **Deliverable**: 4th-order Runge-Kutta ($\text{RK4}$) Lagrangian particle advection engine that runs backward in time (hindcast) to pinpoint the exact latitude/longitude where the dump began, and forward in time (forecast) to trigger coastal beaching warnings.

### (c) AIS Vessel Correlation & Rogue Anomaly Attribution
* **Requirement**: Reconstruct vessel traffic in space and time around the origin release window using historic Automatic Identification System (AIS) data. Filter out irrelevant maritime traffic and score potential suspects based on proximity, trajectory coincidence, and behavioural/kinematic anomalies.
* **Deliverable**: Mathematical multi-factor forensic scoring model that calculates the Closest Point of Approach ($\text{CPA}$), temporal alignment, speed drop anomalies (illicit dumping speed profiles), and vessel class risks, outputting an automated Coast Guard violation dossier.

---

## 4. Why OCEAN-SHIELD is the Winning Solution
1. **End-to-End Autonomous Pipeline**: Unlike fragmented tools that only do SAR detection or only do vessel tracking, OCEAN-SHIELD connects raw satellite pixels directly to a named ship and IMO number in a single execution.
2. **Day/Night All-Weather Reliability**: Built primarily on Sentinel-1 C-Band Synthetic Aperture Radar, it penetrates monsoon clouds, fog, and nighttime conditions.
3. **Defense-Grade Physics**: Replaces naive linear extrapolation with 4th-order Runge-Kutta numerical integration driven by authentic NOAA/US Navy HYCOM 4D NetCDF ocean current vectors and Mackay ADIOS physical weathering kinetics.
4. **Actionable Legal Evidence**: Automatically compiles georeferenced satellite overlays, SOG velocity charts, and provenance metadata into a legally defensible Indian Coast Guard violation report (PDF).
