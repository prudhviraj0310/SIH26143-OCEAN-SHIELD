# OCEAN-SHIELD: Datasets, Provenance & Met-Ocean Sources

## 1. Overview of Data Sources
OCEAN-SHIELD operates on real-world earth observation data, meteorological-oceanographic models, and maritime tracking streams specified directly in NTRO Problem Statement 26143.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-SOURCE DATA PIPELINE                           │
├────────────────────────┬─────────────────────────┬─────────────────────────┤
│    SATELLITE SENSORS   │   MET-OCEAN CURRENTS    │   MARITIME AIS TRACKS   │
├────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • Sentinel-1 SAR (C-Band│ • NOAA / US Navy HYCOM   │ • NOAA MarineCadastre   │
│   VV/VH radar GeoTIFF) │   GOFS 3.1 NetCDF 4D    │   Standard AIS CSV      │
│ • Sentinel-2 Optical EO│ • Global Reanalysis     │ • Real Indian EEZ &     │
│   (10m Multispectral)  │   Ocean Vectors (u, v)  │   Scenario Traffic      │
│ • Zenodo Oil Spill     │ • NOAA GFS / ECMWF 10m  │ • CFAR Radar Targets    │
│   Benchmark Dataset    │   Atmospheric Winds     │   (Non-AIS Dark Ships)  │
└────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 2. Satellite Remote Sensing Data

### A. Synthetic Aperture Radar (SAR) — Sentinel-1
* **Satellite Mission**: Copernicus Sentinel-1A / Sentinel-1B (European Space Agency).
* **Sensor Band & Polarization**: C-Band SAR (~5.405 GHz), VV (Vertical transmit, Vertical receive) and VH polarizations.
* **Why SAR?**: SAR is active microwave imaging. It emits radar pulses and records backscatter reflections. It is completely immune to cloud cover, haze, smoke, and total solar darkness.
* **Physics of Oil on SAR**: Clean ocean waves create surface capillary roughness, which reflects radar back to the satellite (bright grey pixels). Oil floating on seawater damps high-frequency capillary waves (Marangoni damping effect), creating a smooth, mirror-like surface that reflects radar pulses away from the antenna. Consequently, oil spills appear as distinctive dark patches (low backscatter, measured in decibels $\sigma_0 \text{ dB}$).
* **Genuine Files in Codebase**:
  * `datasets/real_sar/2018_09_26.tif`: Genuine 52 MB Sentinel-1 Level-1 Ground Range Detected (GRD) GeoTIFF containing authenticated maritime oil slick signatures.
  * `datasets/real_sar/real_sentinel1_crop_512.png`: Calibrated 512x512 pixel crop for fast inference.
* **Zenodo Benchmark**: Trained and validated against the Zenodo Sentinel-1 SAR Oil Spill Dataset (referenced in the NTRO portal guidelines).

### B. Electro-Optical (EO) — Sentinel-2
* **Satellite Mission**: Copernicus Sentinel-2 (Multispectral Instrument - MSI).
* **Resolution**: 10m ground sampling distance.
* **Role in Pipeline**: Used in daytime clear-sky conditions as a secondary sensor to compute the Normalized Difference Oil Index ($\text{NDOI}$) across NIR and Visible bands, verifying whether a radar dark patch is petroleum crude versus an organic algae bloom or sediment plume.

---

## 3. Oceanographic & Meteorological Data (Met-Ocean)

### A. HYCOM GOFS 3.1 (Hybrid Coordinate Ocean Model)
* **Agency**: US Navy / NOAA / National Centers for Environmental Prediction (NCEP).
* **Format**: NetCDF-4 (`.nc`) 4D scientific array archives (Time, Depth, Latitude, Longitude).
* **Parameters Extracted**:
  * `water_u`: Eastward zonal ocean surface current velocity ($\text{m/s}$).
  * `water_v`: Northward meridional ocean surface current velocity ($\text{m/s}$).
  * `surf_el`: Sea surface height elevation ($\text{m}$).
  * `water_temp`: Sea surface temperature ($^\circ\text{C}$), used in Mackay ADIOS kinetic weathering calculations.
* **Files in Codebase**:
  * `datasets/ocean_met/hycom_real_gulf_kachchh.nc`
  * `datasets/ocean_met/hycom_mumbai_high.nc`
  * `datasets/ocean_met/hycom_great_nicobar.nc`
* **Pragmatic Fallback**: If scientific C-libraries (`libnetcdf`) are absent in minimal hosting environments, the system seamlessly falls back to analytical tidal current models without throwing application crashes.

### B. Atmospheric Wind Forcing
* **Source**: NOAA Global Forecast System (GFS) 10-meter wind vectors ($U_{10}, V_{10}$ in $\text{m/s}$).
* **Wind Leeway Rule**: Oil slicks do not move solely with ocean currents; wind imparts a surface shear force. OCEAN-SHIELD models wind leeway advection at $3.0\%$ to $3.5\%$ of wind velocity with a $0^\circ$ to $15^\circ$ Coriolis deflection angle.

---

## 4. Automatic Identification System (AIS) Vessel Tracking

### A. MarineCadastre.gov Format
NTRO explicitly specified: *"Format of AIS data can be obtained from sample AIS data available at https://marinecadastre.gov/accessais/"*.

OCEAN-SHIELD’s parser (`ais_ingestion.py`) directly ingests standard NOAA / MarineCadastre CSV schema:
1. `MMSI`: Maritime Mobile Service Identity (9-digit unique radio transponder ID).
2. `BaseDateTime`: UTC ISO timestamp of broadcast (`YYYY-MM-DDTHH:MM:SSZ`).
3. `LAT`, `LON`: WGS84 geographic coordinates.
4. `SOG`: Speed Over Ground in knots.
5. `COG`: Course Over Ground in degrees.
6. `Heading`: Ship compass heading in degrees.
7. `VesselName`: Name of the vessel.
8. `IMO`: International Maritime Organization registry number (7 digits).
9. `CallSign`: Radio call sign.
10. `VesselType`: Cargo, Tanker, Fishing, Tug, Passenger, etc.
11. `Status`: Navigation status (Underway using engine, Moored, Anchored, Restricted maneuverability).
12. `Length`, `Width`, `Draft`: Physical vessel dimensions.

### B. Real vs. Synthetic Regional Demarcation
* **NTRO Directive**: The official NTRO Problem Statement explicitly states:
  > *"Real AIS if available may be used else synthetic data can be prepared for the region of oil spill to demonstrate the functioning of the algorithm."*
* **Why Synthetic for Indian EEZ?**: Real-time live Indian coastal AIS data is classified by the Directorate General of Lighthouses and Lightships (DGLL) and Indian Navy / Coast Guard for national defense.
* **OCEAN-SHIELD Implementation**:
  1. Integrates real NOAA MarineCadastre sample data (`datasets/marinecadastre_real_ais.csv`, 212 KB).
  2. Implements realistic scenario traffic for Indian strategic sectors (Gulf of Kachchh SPM terminal, Mumbai High offshore oilfield, Great Nicobar / Malacca Strait chokepoint).
  3. Supports custom drag-and-drop CSV upload for any operator's real fleet data.
  4. Explicitly labels every data point with forensic provenance badges (`DEMO INPUTS` vs `FIELD INPUTS`) so evidence integrity is never misrepresented in court or tribunal.
