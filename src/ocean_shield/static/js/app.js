/**
 * OCEAN-SHIELD Dashboard Application
 * Manages Leaflet GIS map, SAR overlays, Lagrangian particle swarm animation,
 * time-scrubber playback, and AIS vessel attribution telemetry.
 */

class OceanShieldApp {
  constructor() {
    this.activeScenarioId = 'gulf_of_kachchh';
    this.scenarioData = null;
    this.sarResults = null;
    this.driftResults = null;
    this.aisResults = null;
    this.activeSensor = 'sar';
    this.eoResults = null;
    this.customAisVessels = null;
    this.aisProvenance = null;
    this.pendingSarFile = null;
    this.sarProvenance = null;
    this.sceneGeometry = null;
    this.sceneAcquisitionTime = null;

    // Simulation state
    this.currentRelativeTime = 0.0;
    this.isPlaying = false;
    this.playInterval = null;

    // Map & Layers
    this.map = null;
    this.slickLayer = null;
    this.hindcastLayer = null;
    this.forecastLayer = null;
    this.originMarker = null;
    this.particlesLayerGroup = null;
    this.vesselsLayerGroup = null;
    this.sarOverlayLayer = null;

    this.initElements();
    this.initMap();
    this.bindEvents();
    this.loadScenario(this.activeScenarioId);
  }

  initElements() {
    this.scenarioSelector = document.getElementById('scenarioSelector');
    this.btnDetectSAR = document.getElementById('btnDetectSAR');
    this.btnRunDrift = document.getElementById('btnRunDrift');
    this.btnCorrelateAIS = document.getElementById('btnCorrelateAIS');
    this.btnDownloadDossier = document.getElementById('btnDownloadDossier');
    this.srToggle = document.getElementById('srToggle');
    this.modelSelect = document.getElementById('modelSelect');
    this.srActiveBadge = document.getElementById('srActiveBadge');
    this.sarPreviewImg = document.getElementById('sarPreviewImg');

    // Dual Sensor (SAR vs EO) elements
    this.btnSensorSAR = document.getElementById('btnSensorSAR');
    this.btnSensorEO = document.getElementById('btnSensorEO');
    this.sarControlsGroup = document.getElementById('sarControlsGroup');
    this.eoControlsGroup = document.getElementById('eoControlsGroup');
    this.eoLayerSelect = document.getElementById('eoLayerSelect');
    this.eoActiveBadge = document.getElementById('eoActiveBadge');
    this.labelArea = document.getElementById('labelArea');
    this.labelMass = document.getElementById('labelMass');
    this.labelElongation = document.getElementById('labelElongation');
    this.labelConfidence = document.getElementById('labelConfidence');

    // Custom AIS CSV file upload
    this.btnUploadAIS = document.getElementById('btnUploadAIS');
    this.aisFileInput = document.getElementById('aisFileInput');
    this.btnUploadSAR = document.getElementById('btnUploadSAR');
    this.sarFileInput = document.getElementById('sarFileInput');
    this.sarUploadForm = document.getElementById('sarUploadForm');
    this.btnRunUploadedSAR = document.getElementById('btnRunUploadedSAR');
    this.btnConfigureMetOcean = document.getElementById('btnConfigureMetOcean');
    this.metOceanForm = document.getElementById('metOceanForm');
    this.inputModeTag = document.getElementById('inputModeTag');
    this.inputModeCopy = document.getElementById('inputModeCopy');
    this.sourceSummary = document.getElementById('sourceSummary');
    this.oceanDataSourceEl = document.getElementById('oceanDataSource');
    this.aisDataOriginEl = document.getElementById('aisDataOrigin');

    this.timeSlider = document.getElementById('timeSlider');
    this.btnPlayPause = document.getElementById('btnPlayPause');
    this.playIcon = document.getElementById('playIcon');
    this.timePhaseBadge = document.getElementById('timePhaseBadge');
    this.timeDateDisplay = document.getElementById('timeDateDisplay');

    // Metrics
    this.metricArea = document.getElementById('metricArea');
    this.metricMass = document.getElementById('metricMass');
    this.metricElongation = document.getElementById('metricElongation');
    this.metricConfidence = document.getElementById('metricConfidence');
    this.metricCurrent = document.getElementById('metricCurrent');
    this.metricWind = document.getElementById('metricWind');
    this.metricAge = document.getElementById('metricAge');
    this.metricDriftDist = document.getElementById('metricDriftDist');
    this.metricOriginCoords = document.getElementById('metricOriginCoords');
    this.metricBeachingStatus = document.getElementById('metricBeachingStatus');
    this.metricETB = document.getElementById('metricETB');
    this.hazardZoneName = document.getElementById('hazardZoneName');

    // ADIOS Weathering elements
    this.metricEvap = document.getElementById('metricEvap');
    this.metricMousse = document.getElementById('metricMousse');
    this.metricViscosity = document.getElementById('metricViscosity');
    this.metricVolExp = document.getElementById('metricVolExp');
    this.metricWeatheringState = document.getElementById('metricWeatheringState');

    // Suspect card elements
    this.suspectVesselName = document.getElementById('suspectVesselName');
    this.suspectIMO = document.getElementById('suspectIMO');
    this.suspectMMSI = document.getElementById('suspectMMSI');
    this.suspectFlag = document.getElementById('suspectFlag');
    this.suspectType = document.getElementById('suspectType');
    this.suspectScore = document.getElementById('suspectScore');
    this.suspectSummary = document.getElementById('suspectSummary');
    this.vesselTableBody = document.getElementById('vesselTableBody');
    this.vesselCountTag = document.getElementById('vesselCountTag');

    // Anomaly bars
    this.barProx = document.getElementById('barProx');
    this.scoreProx = document.getElementById('scoreProx');
    this.barTime = document.getElementById('barTime');
    this.scoreTime = document.getElementById('scoreTime');
    this.barSpeed = document.getElementById('barSpeed');
    this.scoreSpeed = document.getElementById('scoreSpeed');
    this.barType = document.getElementById('barType');
    this.scoreType = document.getElementById('scoreType');
  }

  initMap() {
    // Initial center around Gulf of Kachchh
    this.map = L.map('tacticalMap', {
      zoomControl: false,
      attributionControl: false
    }).setView([22.465, 69.215], 10);

    L.control.zoom({ position: 'topright' }).addTo(this.map);

    // Multiple basemap layers for judge inspection
    const darkGray = L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      attribution: 'Esri'
    });

    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
      maxNativeZoom: 8,
      attribution: 'Esri World Imagery'
    });

    const googleSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      maxNativeZoom: 11,
      attribution: 'Google Satellite'
    });

    const osmSea = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: 'OpenStreetMap'
    });

    // Default to satellite so judges see real coastlines, islands, and ocean
    satellite.addTo(this.map);

    // Reference labels on top of satellite
    const refLabels = L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      opacity: 0.80
    }).addTo(this.map);

    // Layer control switcher
    const baseMaps = {
      '🛰️ Esri Satellite Imagery': satellite,
      '🌍 Google Earth Satellite': googleSatellite,
      '🌑 Dark Tactical': darkGray,
      '🗺️ OpenStreetMap': osmSea
    };
    L.control.layers(baseMaps, null, { position: 'topright', collapsed: true }).addTo(this.map);

    this.particlesLayerGroup = L.layerGroup().addTo(this.map);
    this.vesselsLayerGroup = L.layerGroup().addTo(this.map);
    this.darkVesselsLayerGroup = L.layerGroup().addTo(this.map);

    // Cursor coordinates readout
    this.map.on('mousemove', (e) => {
      document.getElementById('hudCoords').innerText =
        `${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
    });
  }

  bindEvents() {
    this.scenarioSelector.addEventListener('change', (e) => {
      this.activeScenarioId = e.target.value;
      this.loadScenario(this.activeScenarioId);
    });

    this.btnDetectSAR.addEventListener('click', () => {
      if (this.activeSensor === 'eo') {
        this.runEOAnalysis();
      } else {
        this.runSARAnalysis();
      }
    });
    this.btnRunDrift.addEventListener('click', () => this.runDriftSimulation());
    this.btnCorrelateAIS.addEventListener('click', () => this.runAISCorrelation());
    this.btnDownloadDossier.addEventListener('click', () => this.downloadDossier());

    this.srToggle.addEventListener('change', () => this.toggleSuperResolution());
    if (this.modelSelect) {
      this.modelSelect.addEventListener('change', () => this.runSARAnalysis());
    }

    if (this.btnSensorSAR) {
      this.btnSensorSAR.addEventListener('click', () => this.switchSensor('sar'));
    }
    if (this.btnSensorEO) {
      this.btnSensorEO.addEventListener('click', () => this.switchSensor('eo'));
    }
    if (this.eoLayerSelect) {
      this.eoLayerSelect.addEventListener('change', () => this.renderEOLayer());
    }

    if (this.btnUploadAIS && this.aisFileInput) {
      this.btnUploadAIS.addEventListener('click', () => this.aisFileInput.click());
      this.aisFileInput.addEventListener('change', (e) => this.handleAISUpload(e));
    }

    this.btnUploadSAR?.addEventListener('click', () => this.sarFileInput.click());
    this.sarFileInput?.addEventListener('change', (e) => this.prepareSARUpload(e));
    this.btnRunUploadedSAR?.addEventListener('click', () => this.runUploadedSAR());
    this.btnConfigureMetOcean?.addEventListener('click', () => {
      this.metOceanForm.hidden = !this.metOceanForm.hidden;
    });

    this.timeSlider.addEventListener('input', (e) => {
      this.setTimeOffset(parseFloat(e.target.value));
    });

    this.btnPlayPause.addEventListener('click', () => this.togglePlayback());

    // ── C2 Modern UX: Segmented Tab Switching ──────────────────────────────
    document.querySelectorAll('.dock-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const tabTarget = tab.dataset.tab;
        document.querySelectorAll('.dock-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tabTarget)?.classList.add('active');
      });
    });

    // ── C2 Modern UX: Dock Panel Collapse / Expand ─────────────────────────
    const btnToggleLeft = document.getElementById('btnToggleLeftPanel');
    const panelLeft = document.getElementById('panelLeft');
    if (btnToggleLeft && panelLeft) {
      btnToggleLeft.addEventListener('click', () => {
        panelLeft.classList.toggle('collapsed');
        setTimeout(() => this.map.invalidateSize(), 360);
      });
    }

    const btnToggleRight = document.getElementById('btnToggleRightPanel');
    const panelRight = document.getElementById('panelRight');
    if (btnToggleRight && panelRight) {
      btnToggleRight.addEventListener('click', () => {
        panelRight.classList.toggle('collapsed');
        document.body.classList.toggle('right-collapsed');
        setTimeout(() => this.map.invalidateSize(), 360);
      });
    }
  }

  async loadScenario(scenarioId) {
    this.resetState();
    try {
      const resp = await fetch(`/api/scenario/${scenarioId}`);
      const data = await resp.json();
      this.scenarioData = data.scenario;
      this.pendingSarFile = null;
      this.sarProvenance = null;
      this.customAisVessels = null;
      this.aisProvenance = null;
      this.sceneGeometry = { ...this.scenarioData.center, width: 512, height: 512, pixelSize: 50 };
      this.sceneAcquisitionTime = (this.scenarioData.satellite_metadata.acquisition_time_utc || '')
        .replace(' UTC', 'Z')
        .replace(' ', 'T') || null;
      const toDatetimeLocal = (timestamp) => timestamp ? timestamp.replace(' UTC', '').replace('Z', '').slice(0, 16) : '';
      document.getElementById('sarCenterLat').value = this.scenarioData.center.lat;
      document.getElementById('sarCenterLon').value = this.scenarioData.center.lon;
      document.getElementById('sarPixelSize').value = this.scenarioData.satellite_metadata.pixel_spacing_m || 10;
      document.getElementById('sarAcquisitionUtc').value = toDatetimeLocal(this.sceneAcquisitionTime);
      document.getElementById('timeDateDisplay').innerText =
        this.scenarioData.satellite_metadata.acquisition_time_ist || this.scenarioData.satellite_metadata.acquisition_time_utc || 'Not supplied';
      this.setEvidenceState();

      // Update HUD & Map
      document.getElementById('hudSectorName').innerText = this.scenarioData.region;
      document.getElementById('sarMissionTag').innerText = this.scenarioData.satellite_metadata.mission;
      this.map.setView([this.scenarioData.center.lat, this.scenarioData.center.lon], 10);

      // Met-Ocean cards
      const cond = this.scenarioData.ocean_conditions;
      this.metricCurrent.innerText = `${cond.base_current_u.toFixed(2)}, ${cond.base_current_v.toFixed(2)} m/s`;
      this.metricWind.innerText = `${cond.base_wind_u.toFixed(1)}, ${cond.base_wind_v.toFixed(1)} m/s`;

      const hazard = this.scenarioData.coastline_hazard;
      if (hazard) {
        this.hazardZoneName.innerText = `Threat Zone: ${hazard.coastal_zone_name} (${hazard.distance_to_shore_km} km away)`;
      }

      // SAR Preview
      this.toggleSuperResolution(data);

      // Automatically trigger end-to-end processing so screen is never blank!
      try {
        await this.runSARAnalysis();
        await this.runDriftSimulation();
        await this.runAISCorrelation();
        if (this.btnDetectSAR) this.btnDetectSAR.classList.add('active');
        if (this.btnRunDrift) this.btnRunDrift.classList.add('active');
        if (this.btnCorrelateAIS) this.btnCorrelateAIS.classList.add('active');
        document.getElementById('systemStatusText').innerText = 'WORKSPACE ACTIVE • ALL LEADS RANKED';
      } catch (pipeErr) {
        console.warn('Auto-pipeline warning:', pipeErr);
      }

    } catch (err) {
      console.error('Error loading scenario:', err);
    }
  }

  resetState() {
    this.pausePlayback();
    this.timeSlider.value = 0;
    this.currentRelativeTime = 0;

    if (this.slickLayer) this.map.removeLayer(this.slickLayer);
    if (this.hindcastLayer) this.map.removeLayer(this.hindcastLayer);
    if (this.forecastLayer) this.map.removeLayer(this.forecastLayer);
    if (this.originMarker) this.map.removeLayer(this.originMarker);
    if (this.sarOverlayLayer) this.map.removeLayer(this.sarOverlayLayer);
    this.particlesLayerGroup.clearLayers();
    this.vesselsLayerGroup.clearLayers();
    if (this.darkVesselsLayerGroup) this.darkVesselsLayerGroup.clearLayers();

    // Remove SOG chart if present
    const sogChart = document.getElementById('sogChartContainer');
    if (sogChart) sogChart.remove();
  }

  setEvidenceState() {
    const hasFieldData = Boolean(this.sarProvenance || this.aisProvenance);
    if (this.inputModeTag) {
      this.inputModeTag.className = `tag ${hasFieldData ? 'tag-field' : 'tag-demo'}`;
      this.inputModeTag.innerText = hasFieldData ? 'FIELD INPUTS' : 'DEMO INPUTS';
    }
    if (this.inputModeCopy) {
      this.inputModeCopy.innerText = hasFieldData
        ? 'Uploaded inputs are provenance-labelled. Automated outputs remain analyst-review leads, not findings of liability.'
        : 'This sector is an analyst screening simulation. All suspect rankings are investigative leads requiring official validation.';
    }
    const scenarioSar = this.scenarioData?.satellite_metadata?.data_origin || 'scenario input';
    const sar = this.sarProvenance ? `SAR: ${this.sarProvenance.source_filename}` : `SAR: ${scenarioSar}`;
    const ais = this.aisProvenance ? `AIS: ${this.aisProvenance.source_filename}` : 'AIS: embedded tracks';
    if (this.sourceSummary) {
      this.sourceSummary.innerText = `${sar} • ${ais}`;
    }

    // W3/W2: Safely populate ocean data source and AIS data origin provenance badges
    if (this.scenarioData?.ocean_data_source && this.oceanDataSourceEl) {
      const src = this.scenarioData.ocean_data_source;
      this.oceanDataSourceEl.innerText = `HYCOM: ${src}`;
      const isReal = src.includes('Real') || src.includes('NetCDF');
      this.oceanDataSourceEl.style.color = isReal ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      const dot = this.oceanDataSourceEl.parentElement?.querySelector('.source-dot, .dot-indicator');
      if (dot) dot.style.background = isReal ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }
    if (this.scenarioData?.ais_data_origin && this.aisDataOriginEl) {
      const aisOrigin = this.scenarioData.ais_data_origin;
      this.aisDataOriginEl.innerText = `AIS: ${aisOrigin}`;
      const isReal = aisOrigin.includes('MarineCadastre') || aisOrigin.includes('real');
      this.aisDataOriginEl.style.color = isReal ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      const dot = this.aisDataOriginEl.parentElement?.querySelector('.source-dot, .dot-indicator');
      if (dot) dot.style.background = isReal ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }
  }

  toggleSuperResolution(preloadedData) {
    const isSR = this.srToggle ? this.srToggle.checked : true;
    if (this.srActiveBadge) this.srActiveBadge.style.display = isSR ? 'block' : 'none';

    if (preloadedData && this.sarPreviewImg) {
      const b64 = isSR ? preloadedData.sr_sar_image_base64 : preloadedData.sar_image_base64;
      if (b64) {
        this.sarPreviewImg.src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
      }
    }
  }

  async switchSensor(mode) {
    this.activeSensor = mode;
    if (mode === 'sar') {
      this.btnSensorSAR.style.background = 'rgba(0,242,254,0.2)';
      this.btnSensorSAR.style.borderColor = 'var(--accent-cyan)';
      this.btnSensorSAR.style.color = 'var(--accent-cyan)';
      this.btnSensorEO.style.background = 'transparent';
      this.btnSensorEO.style.borderColor = 'var(--border-subtle)';
      this.btnSensorEO.style.color = 'var(--text-muted)';

      this.sarControlsGroup.style.display = 'flex';
      this.eoControlsGroup.style.display = 'none';
      this.eoActiveBadge.style.display = 'none';
      if (this.srToggle && this.srToggle.checked) this.srActiveBadge.style.display = 'block';

      document.getElementById('sarMissionTag').innerText = this.scenarioData ? this.scenarioData.satellite_metadata.mission : 'SENTINEL-1A VV';
      if (this.labelArea) this.labelArea.innerText = 'Spill Area';
      if (this.labelMass) this.labelMass.innerText = 'Estimated Mass';
      if (this.labelElongation) this.labelElongation.innerText = 'Elongation Ratio';
      if (this.labelConfidence) this.labelConfidence.innerText = 'Oil Confidence';

      await this.runSARAnalysis();
    } else {
      this.btnSensorEO.style.background = 'rgba(245,158,11,0.25)';
      this.btnSensorEO.style.borderColor = 'var(--accent-amber)';
      this.btnSensorEO.style.color = 'var(--accent-amber)';
      this.btnSensorSAR.style.background = 'transparent';
      this.btnSensorSAR.style.borderColor = 'var(--border-subtle)';
      this.btnSensorSAR.style.color = 'var(--text-muted)';

      this.sarControlsGroup.style.display = 'none';
      this.eoControlsGroup.style.display = 'flex';
      this.srActiveBadge.style.display = 'none';
      this.eoActiveBadge.style.display = 'block';

      document.getElementById('sarMissionTag').innerText = 'SENTINEL-2B MSI';
      if (this.labelArea) this.labelArea.innerText = 'Optical Area';
      if (this.labelMass) this.labelMass.innerText = 'Algae Lookalike';
      if (this.labelElongation) this.labelElongation.innerText = 'Mean NDOI';
      if (this.labelConfidence) this.labelConfidence.innerText = 'Optical Confidence';

      await this.runEOAnalysis();
    }
  }

  async runEOAnalysis() {
    document.getElementById('systemStatusText').innerText = 'PROCESSING OPTICAL EO...';
    try {
      const resp = await fetch('/api/analyze-eo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: this.activeScenarioId })
      });
      const data = await resp.json();
      this.eoResults = data;

      const diag = data.diagnostics;
      this.metricArea.innerText = `${diag.area_km2.toFixed(2)} km²`;
      this.metricMass.innerText = diag.lookalike_algae_rejected ? 'REJECTED' : 'CLEAN SEA';
      this.metricElongation.innerText = `+${diag.mean_ndoi.toFixed(3)}`;
      this.metricConfidence.innerText = `${Math.round(diag.confidence * 100)}%`;

      this.renderEOLayer();
      document.getElementById('systemStatusText').innerText = 'EO NDOI COMPUTED';
    } catch (err) {
      console.error('Error running EO analysis:', err);
    }
  }

  renderEOLayer() {
    if (!this.eoResults) return;
    const layer = this.eoLayerSelect ? this.eoLayerSelect.value : 'ndoi_heatmap';
    if (layer === 'true_color') {
      this.sarPreviewImg.src = this.eoResults.rgb_image_base64;
    } else if (layer === 'overlay') {
      this.sarPreviewImg.src = this.eoResults.overlay_base64;
    } else {
      this.sarPreviewImg.src = this.eoResults.ndoi_heatmap_base64;
    }
  }

  async handleAISUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('systemStatusText').innerText = 'INGESTING AIS CSV...';
    try {
      const resp = await fetch('/api/upload-ais-csv', {
        method: 'POST',
        headers: {
          'X-File-Name': file.name,
          'X-Reference-Time-UTC': this.sceneAcquisitionTime || ''
        },
        body: file
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Unable to parse AIS CSV.');
      if (data.status === 'success' && data.vessels) {
        this.customAisVessels = data.vessels;
        this.aisProvenance = data.provenance;
        this.setEvidenceState();
        document.getElementById('systemStatusText').innerText = `AIS READY (${data.vessels_parsed_count} VESSELS)`;
        await this.runAISCorrelation();
      }
    } catch (err) {
      console.error('Error uploading AIS CSV:', err);
      document.getElementById('systemStatusText').innerText = 'AIS INGEST FAILED';
      alert(err.message || 'Error parsing custom AIS CSV file.');
    } finally {
      event.target.value = '';
    }
  }

  prepareSARUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    this.pendingSarFile = file;
    this.sarUploadForm.hidden = false;
    document.getElementById('systemStatusText').innerText = `SAR SELECTED: ${file.name}`;
  }

  async runUploadedSAR() {
    if (!this.pendingSarFile) return;
    document.getElementById('systemStatusText').innerText = 'ANALYZING UPLOADED SAR...';
    try {
      const resp = await fetch('/api/analyze-sar-upload', {
        method: 'POST',
        headers: {
          'X-File-Name': this.pendingSarFile.name,
          'X-Center-Lat': document.getElementById('sarCenterLat').value,
          'X-Center-Lon': document.getElementById('sarCenterLon').value,
          'X-Pixel-Size-M': document.getElementById('sarPixelSize').value,
          'X-Model-Type': this.modelSelect?.value || 'unet',
          'X-Threshold-Offset': '20',
          'X-Acquisition-Time-UTC': document.getElementById('sarAcquisitionUtc').value
            ? `${document.getElementById('sarAcquisitionUtc').value}Z`
            : ''
        },
        body: this.pendingSarFile
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Unable to analyse SAR raster.');
      this.sarProvenance = data.provenance;
      this.sceneAcquisitionTime = data.provenance.acquisition_time_utc || null;
      this.sceneGeometry = {
        lat: data.provenance.scene_center.lat,
        lon: data.provenance.scene_center.lon,
        width: data.provenance.image_shape_px.width,
        height: data.provenance.image_shape_px.height,
        pixelSize: data.provenance.pixel_size_m
      };
      this.setEvidenceState();
      this.sarPreviewImg.src = this.srToggle.checked ? data.super_resolution_base64 : data.segmentation_overlay_base64;
      this.renderSarResponse(data);
      this.sarUploadForm.hidden = true;
      document.getElementById('systemStatusText').innerText = 'UPLOADED SAR SCREENED';
    } catch (err) {
      console.error('Error analysing uploaded SAR:', err);
      document.getElementById('systemStatusText').innerText = 'SAR ANALYSIS FAILED';
      alert(err.message || 'Error analysing uploaded SAR raster.');
    }
  }

  async runSARAnalysis() {
    if (this.pendingSarFile) {
      await this.runUploadedSAR();
      return;
    }
    document.getElementById('systemStatusText').innerText = 'ANALYZING SAR SCENE...';
    try {
      const selectedModel = this.modelSelect ? this.modelSelect.value : 'unet';
      const resp = await fetch('/api/analyze-sar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          use_super_resolution: this.srToggle.checked,
          threshold_offset: 20.0,
          model_type: selectedModel
        })
      });
      const data = await resp.json();
      this.renderSarResponse(data);

      const activeEngine = data.active_engine || (selectedModel === 'unet' ? 'PyTorch U-Net' : 'Adaptive CFAR');
      document.getElementById('systemStatusText').innerText = `SAR SCREENED (${activeEngine.includes('U-Net') ? 'U-NET' : 'CFAR'})`;
      this.updateStepperState(1);
    } catch (err) {
      console.error('Error analyzing SAR:', err);
      document.getElementById('systemStatusText').innerText = 'SAR ANALYSIS FAILED';
    }
  }

  updateStepperState(step) {
    if (this.btnDetectSAR) this.btnDetectSAR.classList.toggle('active', step >= 1);
    if (this.btnRunDrift) this.btnRunDrift.classList.toggle('active', step >= 2);
    if (this.btnCorrelateAIS) this.btnCorrelateAIS.classList.toggle('active', step >= 3);
  }

  renderSarResponse(data) {
      if (!data || !data.sar_results) return;
      this.sarResults = data.sar_results;
      const slick = this.sarResults.primary_slick;
      if (slick) {
        this.metricArea.innerText = `${slick.area_km2.toFixed(2)} km²`;
        this.metricMass.innerText = `${slick.estimated_mass_tonnes.toFixed(1)} T`;
        this.metricElongation.innerText = `${slick.elongation.toFixed(2)}:1`;
        this.metricConfidence.innerText = `${slick.confidence_score}%`;
        this.metricAge.innerText = `${slick.estimated_age_hours || 10.5} h`;

        // Draw Slick Polygon on Map with Glowing Tactical Border
        if (this.slickLayer) this.map.removeLayer(this.slickLayer);
        this.slickLayer = L.geoJSON(slick.polygon_geojson, {
          style: {
            color: '#00f2fe',
            weight: 2.5,
            opacity: 0.95,
            fillColor: '#003366',
            fillOpacity: 0.55
          }
        }).addTo(this.map);

        this.map.fitBounds(this.slickLayer.getBounds(), { padding: [60, 60], maxZoom: 10 });

        // Add slick label
        L.popup({ autoClose: false, closeOnClick: false, className: 'slick-tactical-popup' })
          .setLatLng([slick.centroid.lat, slick.centroid.lon])
          .setContent(`
            <div style="font-size:0.80rem; font-weight:700; color:#00f2fe; margin-bottom:4px; display:flex; align-items:center; gap:6px;">
              <span class="pulse-dot" style="background:#00f2fe;"></span> ${slick.slick_id}
            </div>
            <div style="font-size:0.75rem; color:#f8fafc; font-weight:600; margin-bottom:4px;">${slick.classification}</div>
            <div style="font-size:0.70rem; color:#94a3b8; font-family:var(--font-mono); display:flex; gap:10px;">
              <span>Area: <b style="color:#00f2fe;">${slick.area_km2.toFixed(2)} km²</b></span>
              <span>Mass: <b style="color:#f59e0b;">${slick.estimated_mass_tonnes.toFixed(1)} T</b></span>
            </div>
          `)
          .addTo(this.map);
      }

      // Populate SAR preview image in Detection card
      if (this.sarPreviewImg) {
        const b64 = (this.srToggle && this.srToggle.checked && data.super_resolution_base64)
          ? data.super_resolution_base64
          : (data.segmentation_overlay_base64 || data.sar_image_base64);
        if (b64) {
          this.sarPreviewImg.src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
        }
      }

      // Drape SAR imagery directly onto the map as a geo-referenced overlay
      if (data.segmentation_overlay_base64 && this.sceneGeometry) {
        this.drapeSAROverlay(data.segmentation_overlay_base64);
      }
  }

  drapeSAROverlay(base64Img) {
    // Remove previous SAR overlay
    if (this.sarOverlayLayer) {
      this.map.removeLayer(this.sarOverlayLayer);
    }

    // Calculate bounds from the explicit scene registration, never guessed implicitly.
    const sc = this.sceneGeometry;
    const halfExtentKm = (Math.max(sc.width, sc.height) * sc.pixelSize) / 2000.0;
    const degPerKmLat = 1.0 / 111.32;
    const degPerKmLon = 1.0 / (111.32 * Math.cos(sc.lat * Math.PI / 180));

    const south = sc.lat - halfExtentKm * degPerKmLat;
    const north = sc.lat + halfExtentKm * degPerKmLat;
    const west = sc.lon - halfExtentKm * degPerKmLon;
    const east = sc.lon + halfExtentKm * degPerKmLon;

    const imageBounds = [[south, west], [north, east]];

    this.sarOverlayLayer = L.imageOverlay(`data:image/png;base64,${base64Img}`, imageBounds, {
      opacity: 0.55,
      interactive: false
    }).addTo(this.map);

    // Add opacity slider control
    this.addSAROpacityControl();
  }

  addSAROpacityControl() {
    // Only add once
    if (document.getElementById('sarOpacityCtrl')) return;

    const ctrl = document.createElement('div');
    ctrl.id = 'sarOpacityCtrl';
    ctrl.style.cssText = 'position:absolute; bottom:84px; left:50%; transform:translateX(-50%); z-index:950; background:rgba(9,14,27,0.92); padding:6px 14px; border-radius:20px; border:1px solid rgba(0,242,254,0.35); backdrop-filter:blur(16px); display:flex; align-items:center; gap:10px; box-shadow:0 8px 24px rgba(0,0,0,0.6);';
    ctrl.innerHTML = `
      <span style="font-size:0.68rem; color:#00f2fe; font-family:var(--font-mono); font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">SAR Opacity</span>
      <input type="range" id="sarOpacitySlider" min="0" max="100" value="55" style="width:100px; accent-color:#00f2fe; cursor:pointer;">
    `;
    document.getElementById('tacticalMap').appendChild(ctrl);

    document.getElementById('sarOpacitySlider').addEventListener('input', (e) => {
      if (this.sarOverlayLayer) {
        this.sarOverlayLayer.setOpacity(e.target.value / 100);
      }
    });
  }

  async runDriftSimulation() {
    if (!this.sarResults || !this.sarResults.primary_slick) {
      document.getElementById('systemStatusText').innerText = 'RUNNING SAR DETECTION FIRST...';
      await this.runSARAnalysis();
      if (!this.sarResults || !this.sarResults.primary_slick) return;
    }
    const slick = this.sarResults.primary_slick;

    document.getElementById('systemStatusText').innerText = 'COMPUTING HYDRODYNAMICS...';
    try {
      const optionalNumber = (id) => {
        const value = document.getElementById(id)?.value;
        return value === '' || value === undefined ? null : Number(value);
      };
      const resp = await fetch('/api/simulate-drift', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          slick_lat: slick.centroid.lat,
          slick_lon: slick.centroid.lon,
          slick_age_hours: slick.estimated_age_hours,
          max_lookback_hours: 18.0,
          forecast_hours: 24.0,
          current_u_ms: optionalNumber('currentU'),
          current_v_ms: optionalNumber('currentV'),
          wind_u_ms: optionalNumber('windU'),
          wind_v_ms: optionalNumber('windV')
        })
      });
      const data = await resp.json();
      this.driftResults = data;

      const origin = data.origin_release_point;
      this.metricOriginCoords.innerText = `${origin.lat.toFixed(4)}° N, ${origin.lon.toFixed(4)}° E (T - ${origin.slick_age_hours}h)`;
      this.metricDriftDist.innerText = `${data.total_drift_distance_km.toFixed(1)} km`;

      // ADIOS Physical Oil Weathering metrics
      if (data.weathering_summary) {
        const w = data.weathering_summary;
        if (this.metricEvap) this.metricEvap.innerText = `${w.evaporated_fraction_pct.toFixed(1)}%`;
        if (this.metricMousse) this.metricMousse.innerText = `${w.water_content_mousse_pct.toFixed(1)}%`;
        if (this.metricViscosity) this.metricViscosity.innerText = `${w.dynamic_viscosity_cP.toFixed(0)} cP`;
        if (this.metricVolExp) this.metricVolExp.innerText = `${w.volume_expansion_factor.toFixed(2)}x`;
        if (this.metricWeatheringState) this.metricWeatheringState.innerText = w.weathering_classification;
      }

      // Plot Hindcast (Backward track - dotted red)
      const hindcastPts = data.hindcast_trajectory.map(step => [step.centroid.lat, step.centroid.lon]);
      if (this.hindcastLayer) this.map.removeLayer(this.hindcastLayer);
      this.hindcastLayer = L.polyline(hindcastPts, {
        color: '#ff3366',
        weight: 3,
        dashArray: '5, 8',
        opacity: 0.9
      }).addTo(this.map);

      // Plot Forecast (Forward track - dotted amber)
      const forecastPts = data.forecast_trajectory.map(step => [step.centroid.lat, step.centroid.lon]);
      if (this.forecastLayer) this.map.removeLayer(this.forecastLayer);
      this.forecastLayer = L.polyline(forecastPts, {
        color: '#f59e0b',
        weight: 3,
        dashArray: '6, 6',
        opacity: 0.85
      }).addTo(this.map);

      // Add Origin Marker (x0, y0, t0)
      if (this.originMarker) this.map.removeLayer(this.originMarker);
      const originIcon = L.divIcon({
        className: 'custom-origin-pin',
        html: `<div style="width:16px; height:16px; border-radius:50%; background:#ff3366; border:2px solid #ffffff; box-shadow: 0 0 15px #ff3366;"></div>`,
        iconSize: [16, 16]
      });

      this.originMarker = L.marker([origin.lat, origin.lon], { icon: originIcon })
        .bindPopup(`
          <div style="font-size:0.75rem; font-weight:700; color:#ff3366; margin-bottom:4px; display:flex; align-items:center; gap:6px;">
            <span class="pulse-dot" style="background:#ff3366;"></span> RELEASE CANDIDATE (x₀, y₀)
          </div>
          <div style="font-size:0.72rem; color:#f8fafc; font-weight:600;">Discharge Epicenter</div>
          <div style="font-size:0.68rem; color:#94a3b8; font-family:var(--font-mono); margin-top:2px;">
            Release: <span style="color:#ff3366; font-weight:700;">T - ${origin.slick_age_hours}h</span><br>
            Coords: ${origin.lat.toFixed(4)}° N, ${origin.lon.toFixed(4)}° E
          </div>
        `)
        .addTo(this.map);

      // Update Beaching Hazards
      const beach = data.beaching_warning;
      if (beach.will_beach) {
        this.metricBeachingStatus.innerText = 'IMMINENT INTERCEPT';
        this.metricETB.innerText = `+${beach.estimated_time_to_beach_hours} h`;
      } else {
        this.metricBeachingStatus.innerText = 'OFFSHORE DISPERSION';
        this.metricETB.innerText = 'N/A';
      }

      this.renderParticlesAtTime(0.0);
      document.getElementById('systemStatusText').innerText = 'DRIFT MODEL COMPLETE';
      this.updateStepperState(2);
    } catch (err) {
      console.error('Error running drift simulation:', err);
    }
  }

  async runAISCorrelation() {
    if (!this.driftResults || !this.driftResults.origin_release_point) {
      document.getElementById('systemStatusText').innerText = 'RUNNING DRIFT SIMULATION FIRST...';
      await this.runDriftSimulation();
      if (!this.driftResults || !this.driftResults.origin_release_point) return;
    }
    const origin = this.driftResults.origin_release_point;

    document.getElementById('systemStatusText').innerText = 'CORRELATING AIS TRAFFIC...';
    try {
      const radarTargets = (this.sarResults && this.sarResults.radar_detected_ships) ? this.sarResults.radar_detected_ships : null;
      const resp = await fetch('/api/correlate-ais', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          origin_lat: origin.lat,
          origin_lon: origin.lon,
          origin_time_rel_h: origin.estimated_t0_hours_relative,
          spatial_radius_nm: 30.0,
          temporal_window_h: 5.0,
          hindcast_trajectory: this.driftResults ? this.driftResults.hindcast_trajectory : null,
          radar_targets: radarTargets,
          vessels: this.customAisVessels,
          ais_provenance: this.aisProvenance
        })
      });
      const data = await resp.json();
      this.aisResults = data;

      const culprit = data.primary_culprit;
      if (culprit) {
        this.suspectVesselName.innerText = culprit.vessel_name;
        this.suspectIMO.innerText = culprit.imo;
        this.suspectMMSI.innerText = culprit.mmsi;
        this.suspectFlag.innerText = culprit.flag_state;
        this.suspectType.innerText = culprit.vessel_type;
        this.suspectScore.innerText = `${culprit.composite_suspect_score}%`;
        this.suspectSummary.innerText = `Highest-ranked model lead: ${culprit.attribution_tier}. CPA ${culprit.closest_approach.distance_nm} NM; verify with source records and analyst review.`;

        // Update Anomaly Breakdown — Animated score bars (W4)
        const b = culprit.score_breakdown;
        // Reset all bars to 0 for animation
        this.barProx.style.width = '0%';
        this.barTime.style.width = '0%';
        this.barSpeed.style.width = '0%';
        this.barType.style.width = '0%';

        // Stagger the animations for visual impact
        requestAnimationFrame(() => {
          setTimeout(() => { this.barProx.style.width = `${b.proximity_score}%`; }, 100);
          setTimeout(() => { this.barTime.style.width = `${b.temporal_score}%`; }, 250);
          setTimeout(() => { this.barSpeed.style.width = `${b.speed_anomaly_score}%`; }, 400);
          setTimeout(() => { this.barType.style.width = `${b.vessel_type_score}%`; }, 550);
        });

        this.scoreProx.innerText = `${b.proximity_score} / 100`;
        this.scoreTime.innerText = `${b.temporal_score} / 100`;
        this.scoreSpeed.innerText = `${b.speed_anomaly_score} / 100 (${culprit.kinematics.speed_drop_knots} kts drop)`;
        this.scoreType.innerText = `${b.vessel_type_score} / 100 (${culprit.vessel_type})`;

        // Animate score badge pop-in
        this.suspectScore.classList.remove('score-animate');
        void this.suspectScore.offsetWidth; // force reflow
        this.suspectScore.classList.add('score-animate');
      }

      // Render Table with both AIS suspects and Dark Vessels
      this.renderVesselTable(data.ranked_suspects, data.dark_vessels_detected || []);

      // Plot Vessel Tracks on Map
      this.renderVesselTracks(data.ranked_suspects);

      // Plot Dark Vessels (Non-cooperative radar contacts with disabled AIS)
      this.renderDarkVessels(data.dark_vessels_detected || []);

      // Render SOG Speed-Over-Ground kinematic velocity chart for the culprit
      if (culprit && culprit.full_trajectory) {
        this.renderSOGChart(culprit);
      }

      document.getElementById('systemStatusText').innerText = 'WORKSPACE ACTIVE • ALL LEADS RANKED';
      this.updateStepperState(3);
    } catch (err) {
      console.error('Error correlating AIS:', err);
    }
  }

  renderSOGChart(culprit) {
    // Remove previous chart if any
    const existingChart = document.getElementById('sogChartContainer');
    if (existingChart) existingChart.remove();

    const traj = culprit.full_trajectory;
    if (!traj || traj.length < 2) return;

    const container = document.getElementById('anomalyBreakdownContainer');
    if (!container) return;

    // Build SVG velocity chart
    const chartW = 280, chartH = 100, padL = 35, padR = 10, padT = 8, padB = 22;
    const plotW = chartW - padL - padR;
    const plotH = chartH - padT - padB;

    const times = traj.map(p => p.relative_time_hours);
    const speeds = traj.map(p => p.sog_knots);
    const tMin = Math.min(...times), tMax = Math.max(...times);
    const sMax = Math.max(...speeds) * 1.15;
    const sMin = 0;

    const scaleX = (t) => padL + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const scaleY = (s) => padT + plotH - ((s - sMin) / (sMax - sMin || 1)) * plotH;

    // Highlight the minimum observed speed; this is a review cue, not discharge evidence.
    const minSpeedIdx = speeds.indexOf(Math.min(...speeds));
    const dischargePt = traj[minSpeedIdx];

    // Build polyline path
    const points = traj.map((p, i) => `${scaleX(times[i]).toFixed(1)},${scaleY(speeds[i]).toFixed(1)}`).join(' ');

    // Grid lines
    const gridLines = [5, 10, 15].filter(v => v < sMax).map(v =>
      `<line x1="${padL}" y1="${scaleY(v).toFixed(1)}" x2="${chartW - padR}" y2="${scaleY(v).toFixed(1)}" stroke="rgba(51,65,85,0.5)" stroke-width="0.5" stroke-dasharray="3,3"/>
       <text x="${padL - 4}" y="${scaleY(v).toFixed(1)}" text-anchor="end" fill="#64748b" font-size="7" dominant-baseline="middle">${v}</text>`
    ).join('');

    const svgHtml = `
      <div id="sogChartContainer" style="margin-top:10px; padding:8px; background:rgba(15,23,42,0.7); border:1px solid rgba(51,65,85,0.5); border-radius:6px;">
        <div style="font-size:0.68rem; color:#94a3b8; font-family:var(--text-mono); margin-bottom:4px;">SPEED OVER GROUND (SOG) — ${culprit.vessel_name}</div>
        <svg width="${chartW}" height="${chartH}" viewBox="0 0 ${chartW} ${chartH}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
          <!-- Axes -->
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#475569" stroke-width="1"/>
          <line x1="${padL}" y1="${padT + plotH}" x2="${chartW - padR}" y2="${padT + plotH}" stroke="#475569" stroke-width="1"/>
          ${gridLines}
          <!-- Y axis label -->
          <text x="6" y="${padT + plotH / 2}" fill="#94a3b8" font-size="7" transform="rotate(-90, 6, ${padT + plotH / 2})" text-anchor="middle">kts</text>
          <!-- X axis label -->
          <text x="${padL + plotW / 2}" y="${chartH - 2}" fill="#94a3b8" font-size="7" text-anchor="middle">Relative Time (hours)</text>
          <!-- Speed line -->
          <polyline points="${points}" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linejoin="round"/>
          <!-- Data points -->
          ${traj.map((p, i) => {
            const cx = scaleX(times[i]).toFixed(1);
            const cy = scaleY(speeds[i]).toFixed(1);
            const isDischarge = i === minSpeedIdx;
            return `<circle cx="${cx}" cy="${cy}" r="${isDischarge ? 5 : 2.5}" fill="${isDischarge ? '#ff3366' : '#38bdf8'}" stroke="${isDischarge ? '#fff' : 'none'}" stroke-width="${isDischarge ? 1.5 : 0}"/>` +
              (isDischarge ? `<text x="${cx}" y="${parseFloat(cy) - 8}" fill="#ff3366" font-size="7" text-anchor="middle" font-weight="bold">${p.sog_knots} kts</text>` : '');
          }).join('')}
          <!-- Minimum-speed review cue -->
          <line x1="${scaleX(times[minSpeedIdx]).toFixed(1)}" y1="${padT}" x2="${scaleX(times[minSpeedIdx]).toFixed(1)}" y2="${padT + plotH}" stroke="#ff3366" stroke-width="1" stroke-dasharray="3,2" opacity="0.7"/>
          <text x="${scaleX(times[minSpeedIdx]).toFixed(1)}" y="${padT - 1}" fill="#ff3366" font-size="6" text-anchor="middle">MIN SPEED</text>
        </svg>
      </div>
    `;

    container.insertAdjacentHTML('afterend', svgHtml);
  }

  renderDarkVessels(darkVessels) {
    if (!this.darkVesselsLayerGroup) return;
    this.darkVesselsLayerGroup.clearLayers();

    darkVessels.forEach((dv) => {
      const darkIcon = L.divIcon({
        className: 'dark-vessel-marker',
        html: `
          <div class="dark-vessel-ping">
            <div class="ring"></div>
            <div class="core"></div>
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      const m = L.marker([dv.lat, dv.lon], { icon: darkIcon }).addTo(this.darkVesselsLayerGroup);
      m.bindPopup(`
        <div style="font-family: var(--text-mono); font-size: 0.76rem; min-width: 210px;">
          <b style="color: #ff0044;">⚠️ [RADAR/AIS REVIEW CUE]</b><br>
          <b>Target ID:</b> ${dv.target_id}<br>
          <b>AIS match:</b> <span style="color:#ff3366; font-weight:bold;">${dv.matched_mmsi}</span><br>
          <b>Radar Intensity:</b> ${dv.radar_rcs_mean_db} dB<br>
          <b>Distance to Origin:</b> ${dv.distance_to_spill_origin_nm !== undefined ? dv.distance_to_spill_origin_nm + ' NM' : 'Near Spill'}<br>
          <div style="color: #f59e0b; margin-top: 4px; font-weight: 600;">${dv.threat_classification || 'VERIFY AIS COVERAGE'}</div>
        </div>
      `);
    });
  }

  renderVesselTable(suspects, darkVessels = []) {
    const totalCount = suspects.length + darkVessels.length;
    this.vesselCountTag.innerHTML = darkVessels.length > 0
      ? `${suspects.length} AIS \u2022 ${darkVessels.length} REVIEW`
      : `${suspects.length} VESSELS`;

    if (!suspects.length && !darkVessels.length) {
      this.vesselTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No vessels in corridor.</td></tr>`;
      return;
    }

    let rowsHtml = '';

    // Radar/AIS review cues first; absence of a match is not a disabled transponder.
    darkVessels.forEach(dv => {
      rowsHtml += `
        <tr class="vessel-row dark-vessel-row" style="background: rgba(255, 0, 68, 0.14); border-left: 3px solid #ff0044;">
          <td><b style="color:#ff0044;">⚡ ${dv.target_id}</b><br><span style="font-size:0.65rem; color:#ff6688;">RADAR/AIS REVIEW</span></td>
          <td>RADAR BLIP</td>
          <td>${dv.distance_to_spill_origin_nm !== undefined ? dv.distance_to_spill_origin_nm + ' NM' : 'Near origin'}</td>
          <td style="color:#ff0044; font-weight:bold;">VERIFY COVERAGE</td>
          <td><b style="color:#ff0044;">REVIEW</b></td>
        </tr>
      `;
    });

    suspects.forEach((v, idx) => {
      const isCulprit = idx === 0;
      const rowClass = isCulprit ? 'vessel-row culprit' : 'vessel-row';
      rowsHtml += `
        <tr class="${rowClass}">
          <td><b style="color:${v.flag_color};">${v.vessel_name}</b><br><span style="font-size:0.65rem; color:var(--text-muted);">IMO ${v.imo}</span></td>
          <td>${v.vessel_type}</td>
          <td>${v.closest_approach.distance_nm} NM</td>
          <td style="color:${v.kinematics.speed_drop_knots > 4.0 ? 'var(--accent-crimson)' : 'inherit'};">
            -${v.kinematics.speed_drop_knots} kts
          </td>
          <td><b style="color:${v.flag_color};">${v.composite_suspect_score}%</b></td>
        </tr>
      `;
    });
    this.vesselTableBody.innerHTML = rowsHtml;
  }

  renderVesselTracks(vessels) {
    this.vesselsLayerGroup.clearLayers();

    vessels.forEach((v, idx) => {
      const isCulprit = idx === 0;
      const pts = v.full_trajectory.map(p => [p.lat, p.lon]);
      const color = isCulprit ? '#ff3366' : (v.composite_suspect_score > 50 ? '#ffaa00' : '#05d6a0');

      // Trajectory line
      const poly = L.polyline(pts, {
        color: color,
        weight: isCulprit ? 3.5 : 2,
        opacity: isCulprit ? 0.95 : 0.6
      }).addTo(this.vesselsLayerGroup);

      // Latest position marker
      const latestPt = v.full_trajectory[v.full_trajectory.length - 1];
      const shipMarker = L.circleMarker([latestPt.lat, latestPt.lon], {
        radius: isCulprit ? 7 : 5,
        fillColor: color,
        color: '#ffffff',
        weight: 1.5,
        fillOpacity: 1
      }).bindPopup(`
        <div style="font-size:0.80rem; font-weight:700; color:#f8fafc; margin-bottom:2px;">${v.vessel_name}</div>
        <div style="font-size:0.68rem; color:#94a3b8; margin-bottom:6px;">${v.vessel_type} &bull; MMSI: ${v.mmsi}</div>
        <div style="display:flex; justify-content:space-between; font-size:0.70rem; font-family:var(--font-mono); background:rgba(0,0,0,0.3); padding:4px 6px; border-radius:4px; margin-bottom:3px;">
          <span style="color:#94a3b8;">Lead Score:</span>
          <b style="color:${v.composite_suspect_score > 60 ? '#ef4444' : '#00f2fe'}; font-weight:700;">${v.composite_suspect_score}%</b>
        </div>
        <div style="font-size:0.66rem; color:#64748b; font-family:var(--font-mono);">
          Speed: ${latestPt.sog_knots} kts &bull; Heading: ${latestPt.cog_degrees}°
        </div>
      `);

      this.vesselsLayerGroup.addLayer(shipMarker);
    });
  }

  setTimeOffset(hours) {
    this.currentRelativeTime = hours;
    this.timeSlider.value = hours;

    // Update Header Badge
    if (hours < -0.5) {
      this.timePhaseBadge.className = 'timeline-badge hindcast';
      this.timePhaseBadge.innerText = `T = ${hours.toFixed(1)}h (HINDCAST REVERSE)`;
    } else if (hours > 0.5) {
      this.timePhaseBadge.className = 'timeline-badge forecast';
      this.timePhaseBadge.innerText = `T = +${hours.toFixed(1)}h (FUTURE FORECAST)`;
    } else {
      this.timePhaseBadge.className = 'timeline-badge detect';
      this.timePhaseBadge.innerText = `T = 0.0h (SATELLITE DETECTION PASS)`;
    }

    this.renderParticlesAtTime(hours);
  }

  renderParticlesAtTime(hours) {
    if (!this.driftResults) return;
    this.particlesLayerGroup.clearLayers();

    let trajectory = null;
    if (hours <= 0) {
      trajectory = this.driftResults.hindcast_trajectory;
    } else {
      trajectory = this.driftResults.forecast_trajectory;
    }

    if (!trajectory || !trajectory.length) return;

    // Find closest step in trajectory
    let closestStep = trajectory[0];
    let minDiff = 999;
    for (const step of trajectory) {
      const diff = Math.abs(step.relative_time_hours - hours);
      if (diff < minDiff) {
        minDiff = diff;
        closestStep = step;
      }
    }

    // Render sampled particles
    const color = hours < 0 ? '#ff3366' : (hours > 0 ? '#f59e0b' : '#00f2fe');
    if (closestStep.particles_sample) {
      closestStep.particles_sample.forEach(coord => {
        const pMarker = L.circleMarker([coord[1], coord[0]], {
          radius: 3,
          color: color,
          weight: 1,
          fillColor: color,
          fillOpacity: 0.75
        });
        this.particlesLayerGroup.addLayer(pMarker);
      });
    }
  }

  togglePlayback() {
    if (this.isPlaying) {
      this.pausePlayback();
    } else {
      this.startPlayback();
    }
  }

  startPlayback() {
    this.isPlaying = true;
    this.playIcon.innerHTML = `<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>`;
    this.playInterval = setInterval(() => {
      let nextTime = this.currentRelativeTime + 0.5;
      if (nextTime > 24.0) {
        nextTime = -18.0;
      }
      this.setTimeOffset(nextTime);
    }, 150);
  }

  pausePlayback() {
    this.isPlaying = false;
    if (this.playInterval) clearInterval(this.playInterval);
    this.playIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"/>`;
  }

  async downloadDossier() {
    if (!this.sarResults || !this.driftResults || !this.aisResults) {
      alert('Run SAR, drift, and AIS lead screening before exporting a case summary.');
      return;
    }
    document.getElementById('systemStatusText').innerText = 'BUILDING CASE SUMMARY...';
    try {
      const response = await fetch('/api/export-case-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          sar_results: this.sarResults,
          drift_results: this.driftResults,
          ais_results: this.aisResults,
          evidence_provenance: { sar: this.sarProvenance, ais: this.aisProvenance }
        })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Unable to generate case summary.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'Ocean_Shield_Case_Summary.pdf';
      anchor.click();
      URL.revokeObjectURL(url);
      document.getElementById('systemStatusText').innerText = 'CASE SUMMARY EXPORTED';
    } catch (error) {
      console.error('Case summary export failed:', error);
      document.getElementById('systemStatusText').innerText = 'EXPORT FAILED';
      alert(error.message || 'Unable to generate case summary.');
    }
  }

  // ── W4: Toast notification utility ───────────────────────────────────────
  showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.innerText = message;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.animation = 'toast-slide-out 0.3s ease-in forwards';
      toast.addEventListener('animationend', () => toast.remove());
    }, 4000);
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.oceanShield = new OceanShieldApp();
});
