# OCEAN-SHIELD: Complete UI Elements & Tactical Workflow Guide

## Executive Overview of the User Interface
The **OCEAN-SHIELD** Command and Control (C2) interface is designed as a military-grade, dark-mode tactical maritime cockpit. Built without heavyweight frontend frameworks, it uses ultra-responsive vanilla HTML5, CSS3 glassmorphism, and Leaflet.js running at 60 FPS. The interface balances high-density information display with rapid, frictionless decision-making for Indian Coast Guard (ICG) and National Technical Research Organisation (NTRO) watchstanders.

---

## 1. Top Tactical Header & Classification Bar

| Element | Location | Purpose & Functionality |
| :--- | :--- | :--- |
| **Security Classification Banner** | Very top edge | Displays `SECRET // REL TO IND / NTRO / ICG` in tactical cyan/amber to reflect military mission compliance. |
| **System Identity & Badge** | Top-left | Displays the `OCEAN-SHIELD` insignia, system codename, and build version (`v2.4-PRODUCTION`). |
| **Active Threat Indicator** | Center-left | Dynamic badge showing current incident threat level (`DEFCON 3`, `DEFCON 2`, or `CRITICAL CONTAMINATION`). Changes color from emerald green to warning yellow and pulsing crimson. |
| **Live Geolocation & Cursor Coordinate HUD** | Center | Real-time latitude/longitude readout accurate to 5 decimal places (`DD°MM'SS"N, DD°MM'SS"E`) tracking user cursor position over the GIS map. |
| **System Telemetry & Latency Monitor** | Center-right | Real-time WebSocket / REST heartbeat showing API latency (typically `< 45ms`), backend worker status, and GPU pipeline availability. |
| **Master Operational Clock** | Right | Synchronized Zulu (UTC) and Indian Standard Time (IST) high-precision operational clocks. |
| **Audio Alarm Toggle & Mute Control** | Far right | Controls tactical audio pings (sonar pulses on detection, siren on EEZ breach, click feedback on control engagement). |

---

## 2. Tactical GIS Map Canvas (Central Viewport)

The center of the application is a high-performance Leaflet.js interactive geospatial map capable of rendering multi-layer raster and vector assets simultaneously.

### Map Base Layers (Tile Switcher)
1. **Esri World Imagery (Native High-Resolution Satellite)**:
   - High-resolution optical satellite reconnaissance with zoom levels up to 18.
   - Shows true-color coastlines, coral atolls, barrier islands, and port infrastructure.
2. **CartoDB Dark Matter**:
   - High-contrast tactical night view optimized for low-light command centers.
   - Accentuates fluorescent slick overlays and radar returns.
3. **OpenStreetMap Maritime / Standard**:
   - Standard navigational baseline showing coastal settlements and land topology.
4. **GEBCO Bathymetric Shading**:
   - Ocean depth contours indicating shallow shoals, continental shelf drop-offs, and deep-water shipping lanes.

### Dynamic Vector & Raster Overlays
- **EEZ Boundary (Exclusive Economic Zone)**:
   - 200-nautical-mile maritime boundary polygon clearly marked with dashed cyan lines, establishing Indian sovereign economic jurisdiction.
- **Oil Slick Detection Masks**:
   - Fluorescent magenta/amber polygonal heatmaps delineating the exact oil slick footprint derived from Sentinel-1 SAR and Sentinel-2 Optical processing.
- **Oil Slick Particle Cloud**:
   - Real-time particle markers visualizing the Mackay weathering dispersion and physical droplet spreading.
- **Runge-Kutta (RK4) Drift Trajectory Vector**:
   - Multi-segment trajectory line plotting predicted slick center-of-mass positions at +6h, +12h, +24h, +48h, and +72h intervals.
   - Interactive milestone nodes display timestamp, predicted area, and distance to coast when clicked.
- **3-km Exclusion & Vulnerability Buffers**:
   - Translucent red warning rings surrounding protected marine biospheres (e.g., Gulf of Mannar Marine National Park, coral reefs, turtle nesting beaches).
- **AIS Vessel Targets**:
   - Interactive ship icons placed at real-time/historical coordinates.
   - **Green Triangle**: AIS-compliant cargo/tanker vessel on legal transit.
   - **Yellow Diamond**: Vessel with suspicious route deviations or speed anomalies.
   - **Flashing Red Skull / Hexagon**: Primary Suspect dark ship (AIS disabled, spoofed MMSI, or closest point of approach matched to slick origin).

---

## 3. Left Control Panel: Mission Execution Cockpit

The left sidebar houses the operational controls that allow an officer to run end-to-end detection, drift forecasting, and vessel interdiction.

### 1. Scenario Quick-Selector
A dropdown menu loaded with pre-configured tactical scenarios across high-risk Indian maritime zones:
- **Gulf of Mannar**: Heavy tanker traffic near the sensitive biosphere reserve and international maritime boundary line (IMBL).
- **Bombay High Offshore**: Offshore oil extraction platforms and crude carrier transit routes in the Arabian Sea.
- **Palk Strait**: Shallow passage between India and Sri Lanka prone to illicit bunker transfers.
- **Andaman & Nicobar Channel**: Choke point near the Malacca Strait with dense international supertanker traffic.
- **Lakshadweep Sea**: Pristine atolls on the West-East container shipping highway.
- **Custom Upload Mode**: Allows dragging and dropping user-supplied SAR GeoTIFFs, Optical imagery, or AIS CSV dumps.

### 2. The 4-Step Operational Pipeline Buttons

```
[ Step 1: Detect & Ingest ] ➔ [ Step 2: Forecast & Drift ] ➔ [ Step 3: Intercept & Correlate ] ➔ [ Step 4: Dispatch & Export ]
```

1. **Step 1: Detect & Ingest (SAR / Optical Analysis)**:
   - Triggers ingestion of Sentinel-1 / Sentinel-2 imagery.
   - Executes ESPCN 2X super-resolution to sharpen raw sensor data.
   - Runs PyTorch U-Net segmentation to isolate oil slicks from natural ocean look-alikes.
   - Generates geographic GeoJSON polygons and computes surface area in km² and spill volume in metric tons.
2. **Step 2: Forecast & Drift (HYCOM + GFS Ocean Physics)**:
   - Fetches 4D ocean current vectors ($u, v$) from HYCOM and wind vectors ($U_{10}, V_{10}$) from NOAA GFS.
   - Integrates the 4th-Order Runge-Kutta (RK4) advection differential equations.
   - Simulates Mackay ADIOS evaporative and emulsification kinetics.
   - Computes Time-to-Shoreline (ETA) and flags critical coastal landfall threats.
3. **Step 3: Intercept & Correlate (AIS Forensic Dark Ship Hunter)**:
   - Scans MarineCadastre AIS positional records within a 50 km bounding box.
   - Reconstructs vessel tracks back to the time of estimated spill origin ($t_0$).
   - Calculates Closest Point of Approach (CPA) and spatial-temporal overlap.
   - Ranks suspect vessels using the 4-factor composite risk scoring algorithm and isolates the culprit.
4. **Step 4: Dispatch & Export (Actionable Tactical PDF)**:
   - Compiles full forensic findings into an official NTRO/ICG Incident Report PDF.
   - Formulates automated interception vectors for Indian Coast Guard Dornier-228 patrol aircraft and Offshore Patrol Vessels (OPVs).

> **Smart Auto-Cascade Execution**: If a watchstander clicks "Step 3: Intercept & Correlate" directly, the backend automatically runs Step 1 and Step 2 in seamless sequence, preventing pipeline errors or incomplete state.

---

## 4. Right Analytics Panel: Telemetry & Forensics HUD

The right sidebar provides deep scientific telemetry and legal forensic evidence required for enforcement.

### 1. Environmental & Spill Telemetry Cards
- **Estimated Oil Volume**: Total crude discharge in metric tons ($m^3$ and bbls).
- **Active Surface Footprint**: Surface slick area measured in square kilometers ($\text{km}^2$) and hectares.
- **Current Drift Velocity**: Net drift speed in knots and heading in true degrees ($\circ \text{True}$).
- **Time-to-Shoreline (Landfall ETA)**: High-visibility countdown timer indicating hours and minutes until the oil slick contacts coastline or sensitive marine sanctuaries.

### 2. ADIOS Chemical Weathering Breakdown (Visual Radial / Progress Bars)
- **Evaporated Fraction ($\%$)**: Volatile light ends lost to the atmosphere within the first 24–48 hours.
- **Emulsified Fraction ($\%$)**: Formation of dense "chocolate mousse" water-in-oil emulsion, increasing viscosity.
- **Natural Dispersion ($\%$)**: Entrainment of microscopic oil droplets into the water column driven by wave action.
- **Heavy Residual Core ($\%$)**: Tar-like asphaltic residue persisting on the surface, posing severe long-term hazards.

### 3. Dark Ship Suspect Dossier (Culprit Identification Card)
- **Vessel Identification**: Name, Maritime Mobile Service Identity (MMSI), IMO Number, and Flag State.
- **Vessel Type & Dimensions**: Crude Tanker, Chemical Carrier, Container Ship, or Bulk Carrier (Length, Beam, Gross Tonnage).
- **Closest Point of Approach (CPA)**: Exact distance in nautical miles between the vessel's reconstructed path and the spill ground-zero.
- **AIS Tampering / Dark Activity**: Flags unexplained AIS transmitter shutoffs, discontinuous GPS jumps, or speed anomalies during discharge.
- **Composite Risk Score**: High-contrast score ($0 - 100\%$) indicating mathematical probability of culpability.
- **Action Button**: `Flag to Maritime Operations Center (MOC)`.

### 4. Tactical Event Log Feed
- Chronological timestamped event feed detailing:
  - Satellite pass ingestion confirmation
  - Detection confidence metrics
  - EEZ sovereign violation alerts
  - Landfall proximity warnings

---

## 5. Bottom Command Drawer & Mission Export Bar

- **Generate Mission Intelligence PDF**: One-click generation of the official 3-page forensic dossier containing satellite imagery, drift plots, weathering graphs, suspect vessel AIS tables, and commanding officer sign-off blocks.
- **Terminal & System Diagnostic Drawer**: Expandable collapsible tray displaying live Python API logs, PyTorch tensor memory allocation, and database queries.
- **Unit Dispatch Modal**: Allows the watchstander to select deployed Coast Guard assets (e.g., *ICGS Samudra Prahari*, *ICG Dornier CG-782*) and transmit interception coordinates directly to tactical datalinks.

---

## 6. End-to-End User Journey (The 60-Second Demo Flow)

1. **Open Application**: Officer lands on the C2 dashboard; map initializes over the Gulf of Mannar in high-resolution satellite view.
2. **Select Scenario**: Choose `Gulf of Mannar - Biosphere Threat` from the dropdown.
3. **Execute Step 1**: Click `Detect & Ingest`. High-resolution SAR imagery loads, showing dark slick patches. The U-Net mask renders a glowing fluorescent polygon over the spill. Area and volume metrics update instantly.
4. **Execute Step 2**: Click `Forecast & Drift`. The RK4 trajectory line animates across the water. ADIOS weathering bars show 28% evaporated and 18% emulsified. Landfall alert triggers: *Warning: Coral Reef Impact in 14.2 Hours*.
5. **Execute Step 3**: Click `Intercept & Correlate`. AIS vessel paths appear. The system isolates the culprit tanker: *MT Ocean Vanguard (MMSI: 419001284)* with a 94.8% culprit risk score and CPA of 0.42 nm.
6. **Generate PDF**: Click `Export Mission Intelligence PDF`. A formal, forensic intelligence brief opens, ready for naval interdiction.
