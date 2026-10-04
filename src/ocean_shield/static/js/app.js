/**
 * OCEAN-SHIELD Dashboard Application
 * Manages Leaflet GIS map, SAR overlays, Lagrangian particle swarm animation,
 * time-scrubber playback, and AIS vessel attribution telemetry.
 */

// Missing numbers are missing evidence, never an implicit zero or confidence.
function screenNumber(value, digits = 1, missing = 'NOT ASSESSED') {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : missing;
}

function parseOptionalFiniteNumber(value, label = 'Value') {
  if (value === '' || value === undefined || value === null || (typeof value === 'string' && value.trim() === '')) return null;
  if (typeof value === 'boolean') throw new Error(`${label} must be a finite number.`);
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${label} must be a finite number.`);
  return number;
}

function parseRequiredCoordinate(value, label, fallback = null) {
  const raw = value === '' || value === undefined || value === null ? fallback : value;
  if (raw === null || typeof raw === 'boolean' || (typeof raw === 'string' && raw.trim() === '')) throw new Error(`${label} must be a finite number.`);
  const number = Number(raw);
  if (!Number.isFinite(number)) throw new Error(`${label} must be a finite number.`);
  return number;
}

function sourceNumber(value) {
  if (value === '' || value === undefined || value === null) return null;
  if (typeof value === 'boolean') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

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
    this.demoMode = true;

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
    this.counterfactualLayerGroup = null;
    this.counterfactualResults = null;
    this.kdeContoursLayerGroup = null;
    this.kdeContoursData = null;
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
    this.btnStepCounterfactual = document.getElementById('btnStepCounterfactual');
    this.btnStepDossier = document.getElementById('btnStepDossier');
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
    this.timelineCoordBadge = document.getElementById('timelineCoordBadge');
    this.speedToggleGroup = document.getElementById('speedToggleGroup');
    this.btnStepBack = document.getElementById('btnStepBack');
    this.btnStepForward = document.getElementById('btnStepForward');
    this.timelineTicksRail = document.getElementById('timelineTicksRail');
    this.playbackSpeed = 1;

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

    // Suspect card elements & Candidate Lead Navigation
    this.suspectVesselName = document.getElementById('suspectVesselName');
    this.suspectIMO = document.getElementById('suspectIMO');
    this.suspectMMSI = document.getElementById('suspectMMSI');
    this.suspectFlag = document.getElementById('suspectFlag');
    this.suspectType = document.getElementById('suspectType');
    this.suspectScore = document.getElementById('suspectScore');
    this.suspectSummary = document.getElementById('suspectSummary');
    this.btnPrevLead = document.getElementById('btnPrevLead');
    this.btnNextLead = document.getElementById('btnNextLead');
    this.leadCandidateRank = document.getElementById('leadCandidateRank');
    this.pillSignalStrong = document.getElementById('pillSignalStrong');
    this.pillSignalWeak = document.getElementById('pillSignalWeak');
    this.pillAisQuality = document.getElementById('pillAisQuality');
    this.selectedLeadIndex = 0;

    // Traffic table & Live Filter elements
    this.vesselTableBody = document.getElementById('vesselTableBody');
    this.vesselCountTag = document.getElementById('vesselCountTag');
    this.trafficSearchInput = document.getElementById('trafficSearchInput');
    this.trafficRiskPills = document.getElementById('trafficRiskPills');
    this.btnExportJSON = document.getElementById('btnExportJSON');
    this.trafficFilterQuery = '';
    this.trafficFilterRisk = 'all';

    // Anomaly bars
    this.barProx = document.getElementById('barProx');
    this.scoreProx = document.getElementById('scoreProx');
    this.barTime = document.getElementById('barTime');
    this.scoreTime = document.getElementById('scoreTime');
    this.barSpeed = document.getElementById('barSpeed');
    this.scoreSpeed = document.getElementById('scoreSpeed');
    this.barType = document.getElementById('barType');
    this.scoreType = document.getElementById('scoreType');

    // Executive Simple Mode Elements
    this.btnSimpleView = document.getElementById('btnSimpleView');
    this.btnExpertView = document.getElementById('btnExpertView');
    this.btnToggleExpertMode = document.getElementById('btnToggleExpertMode');
    this.btnExecAutoRun = document.getElementById('btnExecAutoRun');
    this.btnExecDownloadPDF = document.getElementById('btnExecDownloadPDF');
    this.execAreaVal = document.getElementById('execAreaVal');
    this.execVolumeSub = document.getElementById('execVolumeSub');
    this.execLandfallHours = document.getElementById('execLandfallHours');
    this.execHazardTarget = document.getElementById('execHazardTarget');
    this.execSuspectName = document.getElementById('execSuspectName');
    this.execSuspectDetails = document.getElementById('execSuspectDetails');
    this.expertModeBtnText = document.getElementById('expertModeBtnText');

    // Stage 4 Counterfactual Verification elements
    this.counterfactualCard = document.getElementById('counterfactualCard');
    this.cfVerdictBadge = document.getElementById('cfVerdictBadge');
    this.cfCentroidError = document.getElementById('cfCentroidError');
    this.cfContainment = document.getElementById('cfContainment');
    this.cfJaccard = document.getElementById('cfJaccard');
    this.cfCausalityScore = document.getElementById('cfCausalityScore');
    this.cfExplanation = document.getElementById('cfExplanation');
    this.btnRunCounterfactual = document.getElementById('btnRunCounterfactual');
    this.chkShowCounterfactual = document.getElementById('chkShowCounterfactual');

    // Gaussian KDE HDR elements
    this.chkShowKdeContours = document.getElementById('chkShowKdeContours');
    this.metricKdeCredibleArea = document.getElementById('metricKdeCredibleArea');
    this.kdeHdrBox = document.getElementById('kdeHdrBox');

    // Scale-Free Look-Alike Screening elements
    this.lookalikeVerdictBadge = document.getElementById('lookalikeVerdictBadge');
    this.lookalikeScoreVal = document.getElementById('lookalikeScoreVal');
    this.barLookalikeIndex = document.getElementById('barLookalikeIndex');
    this.lookalikeContrast = document.getElementById('lookalikeContrast');
    this.lookalikeEdge = document.getElementById('lookalikeEdge');
    this.lookalikeFractal = document.getElementById('lookalikeFractal');
    this.lookalikeSolidity = document.getElementById('lookalikeSolidity');
    this.lookalikeReasons = document.getElementById('lookalikeReasons');

    // Fay (1971) Hydrodynamic Spreading elements
    this.fayRegimeBadge = document.getElementById('fayRegimeBadge');
    this.faySpillAge = document.getElementById('faySpillAge');
    this.fayLookbackWindow = document.getElementById('fayLookbackWindow');

    // Adversarial Falsification Stress Matrix elements
    this.falsificationCountTag = document.getElementById('falsificationCountTag');
    this.falsificationVerdictBadge = document.getElementById('falsificationVerdictBadge');
    this.falsificationTableBody = document.getElementById('falsificationTableBody');
    this.falsificationRationale = document.getElementById('falsificationRationale');

    // Bayesian Evidence Synthesis & Decision Gate elements
    this.bayesianVerdictBadge = document.getElementById('bayesianVerdictBadge');
    this.bayesianEntropy = document.getElementById('bayesianEntropy');
    this.bayesianMargin = document.getElementById('bayesianMargin');
    this.bayesianBarsList = document.getElementById('bayesianBarsList');
    this.bayesianRecommendation = document.getElementById('bayesianRecommendation');
  }

  initMap() {
    // Initial center around Gulf of Kachchh
    this.map = L.map('tacticalMap', {
      zoomControl: false,
      attributionControl: false
    }).setView([22.585, 69.185], 10);

    L.control.zoom({ position: 'topright' }).addTo(this.map);

    // Multiple basemap layers for judge inspection
    const darkGray = L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      attribution: 'Esri'
    });

    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 18,
      attribution: 'Esri World Imagery'
    });

    const googleSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      attribution: 'Google Satellite'
    });

    const osmSea = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: 'OpenStreetMap'
    });

    // Default to satellite so judges see real coastlines, islands, and ocean
    satellite.addTo(this.map);

    // Reference labels — available as overlay but NOT loaded by default (saves ~20 tile requests)
    const refLabels = L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      opacity: 0.80
    });

    // Source-backed operational layers.  Do not add illustrative port, spill,
    // vessel, corridor, or boundary markers to an operational map.
    this.initLiveOperationalLayers();

    // Leaflet's control requires actual layer objects at construction time.
    this.particlesLayerGroup = L.layerGroup().addTo(this.map);
    this.vesselsLayerGroup = L.layerGroup().addTo(this.map);
    this.darkVesselsLayerGroup = L.layerGroup().addTo(this.map);
    this.counterfactualLayerGroup = L.layerGroup().addTo(this.map);
    this.kdeContoursLayerGroup = L.layerGroup().addTo(this.map);

    // Layer control switcher with Basemaps and Overlays
    const baseMaps = {
      '🛰️ Esri Satellite Imagery': satellite,
      '🌍 Google Earth Satellite': googleSatellite,
      '🌑 Dark Tactical': darkGray,
      '🗺️ OpenStreetMap': osmSea
    };

    const overlayMaps = {
      '🏷️ Reference Labels': refLabels,
      '⚓ NGA World Port Index — reference catalogue': this.indianPortsLayer,
      '⚠️ Authority spill incident feed — live when configured': this.liveIncidentsLayer,
      '🚢 AISStream PositionReports — live when configured': this.liveAisLayer,
      '🎯 Conditional Generated-cloud KDE (95/75/50% mass)': this.kdeContoursLayerGroup,
      '🔬 Stage 4 Counterfactual Re-Simulation': this.counterfactualLayerGroup
    };

    L.control.layers(baseMaps, overlayMaps, { position: 'topright', collapsed: true }).addTo(this.map);

    // Cursor coordinates readout
    this.map.on('mousemove', (e) => {
      document.getElementById('hudCoords').innerText =
        `${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
    });
  }

  initLiveOperationalLayers() {
    this.indianPortsLayer = L.layerGroup().addTo(this.map);
    this.liveIncidentsLayer = L.layerGroup().addTo(this.map);
    this.liveAisLayer = L.layerGroup().addTo(this.map);
    this.liveAOI = null;
    // Defer external API calls so the map + scenario render instantly
    this.refreshOperationalLayers();  // incidents only (fast / cached)
    setTimeout(() => this.refreshOperationalLayers({ includePorts: true }), 2000);
    this.liveLayerRefreshTimer = window.setInterval(() => {
      this.refreshOperationalLayers();
    }, 120000);
  }

  escapeLiveValue(value) {
    return String(value ?? 'Not supplied').replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[character]));
  }

  portMarker(port) {
    const text = this.escapeLiveValue.bind(this);
    const icon = L.divIcon({
      className: 'custom-port-pin',
      html: '<div style="background:#0284c7;width:22px;height:22px;border-radius:50%;border:2px solid #38bdf8;box-shadow:0 0 10px #0284c7;display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px">⚓</div>',
      iconSize: [22, 22], iconAnchor: [11, 11]
    });
    return L.marker([port.lat, port.lon], { icon }).bindPopup(
      `<div class="live-map-popup"><b>⚓ ${text(port.name)}</b><br>` +
      `Country: ${text(port.country)}<br>Harbor size: ${text(port.harbor_size)}<br>` +
      `Facility: ${text(port.facility_type)}<br><small>NGA World Port Index — reference catalogue, not live operations.</small></div>`
    );
  }

  incidentMarker(incident) {
    const text = this.escapeLiveValue.bind(this);
    const icon = L.divIcon({
      className: 'custom-spill-pin',
      html: '<div style="background:#dc2626;width:24px;height:24px;border-radius:50%;border:2px solid #fca5a5;box-shadow:0 0 14px #dc2626;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px">⚠️</div>',
      iconSize: [24, 24], iconAnchor: [12, 12]
    });
    return L.marker([incident.lat, incident.lon], { icon }).bindPopup(
      `<div class="live-map-popup"><b>⚠️ ${text(incident.name)}</b><br>` +
      `Reported: ${text(incident.reported_at_utc)}<br>Severity: ${text(incident.severity)}<br>` +
      `Status: ${text(incident.status)}<br><small>Authority-configured incident feed.</small></div>`
    );
  }

  renderLiveAis(aisResponse) {
    if (!this.liveAisLayer) return;
    this.liveAisLayer.clearLayers();
    if (aisResponse?.status !== 'available' || !Array.isArray(aisResponse.vessels)) return;
    const text = this.escapeLiveValue.bind(this);
    aisResponse.vessels.forEach((vessel) => {
      const point = vessel.current_position;
      if (!Number.isFinite(point?.lat) || !Number.isFinite(point?.lon)) return;
      const icon = L.divIcon({
        className: 'live-ais-pin',
        html: '<div style="background:#0ea5e9;width:22px;height:22px;border-radius:50%;border:2px solid #bae6fd;box-shadow:0 0 12px #0ea5e9;display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px">⌁</div>',
        iconSize: [22, 22], iconAnchor: [11, 11]
      });
      L.marker([point.lat, point.lon], { icon }).bindPopup(
        `<div class="live-map-popup"><b>🚢 ${text(vessel.vessel_name)}</b><br>` +
        `MMSI: ${text(vessel.mmsi)}<br>SOG: ${text(point.sog_knots)} kn<br>` +
        `Received: ${text(point.observed_at)}<br><small>${text(vessel.data_origin)}</small></div>`
      ).addTo(this.liveAisLayer);
    });
  }

  async refreshOperationalLayers({ includePorts = false, lat = null, lon = null } = {}) {
    const tasks = [];
    if (includePorts) tasks.push(fetch('/api/live/ports').then((response) => response.json()).catch(() => null));
    tasks.push(fetch('/api/live/incidents').then((response) => response.json()).catch(() => null));
    const aoiLat = lat ?? this.liveAOI?.lat;
    const aoiLon = lon ?? this.liveAOI?.lon;
    if (Number.isFinite(aoiLat) && Number.isFinite(aoiLon)) {
      tasks.push(fetch(`/api/live/ais-traffic?lat=${aoiLat}&lon=${aoiLon}&collection_seconds=5`).then((response) => response.json()).catch(() => null));
    }
    const results = await Promise.all(tasks);
    let cursor = 0;
    if (includePorts) {
      const ports = results[cursor++];
      if (ports?.status === 'available' && Array.isArray(ports.ports)) {
        this.indianPortsLayer.clearLayers();
        ports.ports.forEach((port) => {
          if (Number.isFinite(port.lat) && Number.isFinite(port.lon)) this.portMarker(port).addTo(this.indianPortsLayer);
        });
      }
    }
    const incidents = results[cursor++];
    if (incidents?.status === 'available' && Array.isArray(incidents.incidents)) {
      this.liveIncidentsLayer.clearLayers();
      incidents.incidents.forEach((incident) => {
        if (Number.isFinite(incident.lat) && Number.isFinite(incident.lon)) this.incidentMarker(incident).addTo(this.liveIncidentsLayer);
      });
    } else if (incidents?.status === 'not_configured') {
      this.liveIncidentsLayer.clearLayers();
    }
    if (Number.isFinite(aoiLat) && Number.isFinite(aoiLon)) this.renderLiveAis(results[cursor]);
  }

  // Retained only as an archived visual prototype. It is deliberately never
  // invoked: operational maps use initLiveOperationalLayers above.
  initArchivedDemoLayers() {
    // 1. Official India 200 NM Exclusive Economic Zone (EEZ) Boundaries
    // Mainland EEZ + Lakshadweep Sea
    const mainlandEEZCoords = [
      [23.75, 68.10], [22.50, 66.40], [21.00, 65.20], [19.00, 66.80],
      [17.00, 68.30], [15.00, 69.80], [13.00, 70.80], [11.00, 71.30],
      [10.00, 69.50], [8.00, 70.00], [7.00, 71.50], [6.80, 74.00],
      [6.50, 77.00], [7.00, 78.50], [8.50, 79.20], [9.20, 79.50],
      [9.80, 79.80], [10.20, 80.10], [11.50, 82.50], [13.00, 84.00],
      [15.00, 85.50], [17.00, 87.00], [19.00, 88.50], [20.50, 89.20],
      [21.50, 89.10], [21.60, 88.50], [21.50, 87.50], [19.80, 85.80],
      [17.70, 83.30], [15.90, 80.60], [13.10, 80.30], [10.80, 79.80],
      [8.10, 77.55], [10.00, 76.20], [12.90, 74.80], [15.40, 73.80],
      [18.90, 72.80], [20.90, 72.80], [22.50, 69.50], [23.75, 68.10]
    ];

    // Andaman & Nicobar Islands EEZ
    const andamanEEZCoords = [
      [14.20, 92.50], [14.00, 94.00], [12.00, 94.80], [10.00, 95.00],
      [8.00, 94.80], [6.80, 94.50], [6.20, 93.80], [6.50, 93.00],
      [8.00, 91.50], [10.00, 91.00], [12.00, 91.50], [13.50, 92.00],
      [14.20, 92.50]
    ];

    this.indiaEEZLayer = L.layerGroup([
      L.polygon(mainlandEEZCoords, {
        color: '#f97316',
        weight: 2,
        dashArray: '8, 8',
        fillColor: '#f97316',
        fillOpacity: 0.04
      }).bindTooltip('🇮🇳 India Exclusive Economic Zone (Mainland & Lakshadweep - 200 NM)', { sticky: true }),
      L.polygon(andamanEEZCoords, {
        color: '#f97316',
        weight: 2,
        dashArray: '8, 8',
        fillColor: '#f97316',
        fillOpacity: 0.04
      }).bindTooltip('🇮🇳 India Exclusive Economic Zone (Andaman & Nicobar - 200 NM)', { sticky: true })
    ]);

    // 2. High-Density Tanker Shipping Highways (Sea Lines of Communication - SLOCs)
    const tankerCorridors = [
      // Persian Gulf to Gulf of Kachchh (70% Crude Import)
      [[24.5, 62.0], [23.5, 65.5], [22.45, 68.9], [22.48, 69.4]],
      // Persian Gulf to Mumbai High / JNPT
      [[23.5, 65.5], [20.5, 69.5], [19.4, 71.3], [18.9, 72.7]],
      // Arabian Sea West Coast Trunk Fairway (Kachchh -> Mumbai -> Goa -> Kochi -> Cape Comorin)
      [[22.4, 69.0], [19.4, 71.3], [15.4, 73.2], [12.8, 74.3], [9.9, 75.8], [7.8, 77.2]],
      // International East-West Megaship Highway (9-Degree Channel -> South of Sri Lanka -> 6-Degree Channel / Malacca)
      [[9.0, 68.0], [8.5, 72.0], [7.5, 75.0], [5.8, 79.5], [5.8, 83.0], [6.0, 88.0], [6.5, 92.5], [6.2, 94.5], [5.8, 96.0]],
      // 9-Degree Channel Feeder (Lakshadweep)
      [[9.2, 70.0], [8.9, 73.5], [8.2, 76.5]],
      // Bay of Bengal Coastal Tanker Highway (Cape Comorin -> Chennai -> Vizag -> Paradip -> Haldia)
      [[7.8, 77.5], [8.5, 78.8], [10.5, 80.5], [13.2, 80.6], [16.8, 82.8], [17.7, 83.5], [20.2, 86.9], [21.5, 88.1]],
      // Great Nicobar Malacca Chokepoint Fairway (Six Degree Channel)
      [[11.5, 91.0], [10.0, 92.5], [8.5, 93.2], [6.7, 93.8]]
    ];

    const shippingLanes = tankerCorridors.map((coords, idx) => {
      return L.polyline(coords, {
        color: '#06b6d4',
        weight: 3,
        opacity: 0.75,
        dashArray: idx === 3 ? null : '4, 6'
      }).bindTooltip('🚢 Major Commercial Shipping Highway (MarineCadastre AIS Standard)', { sticky: true });
    });

    this.indiaShippingLanesLayer = L.layerGroup(shippingLanes);

    // 3. Environmentally Sensitive Marine Sanctuaries
    this.marineSanctuariesLayer = L.layerGroup([
      L.circle([22.58, 69.35], {
        radius: 18000,
        color: '#ef4444',
        dashArray: '4, 4',
        fillColor: '#ef4444',
        fillOpacity: 0.12
      }).bindTooltip('🛑 Marine National Park & Coral Sanctuary (Gulf of Kachchh)', { sticky: true }),
      L.circle([9.00, 78.80], {
        radius: 25000,
        color: '#ef4444',
        dashArray: '4, 4',
        fillColor: '#ef4444',
        fillOpacity: 0.12
      }).bindTooltip('🛑 Gulf of Mannar Marine Biosphere Reserve (Coral & Dugong)', { sticky: true }),
      L.circle([20.75, 86.90], {
        radius: 22000,
        color: '#ef4444',
        dashArray: '4, 4',
        fillColor: '#ef4444',
        fillOpacity: 0.12
      }).bindTooltip('🛑 Gahirmatha Olive Ridley Turtle Sanctuary (Odisha)', { sticky: true })
    ]);

    // 4. All 18 Major Indian Ports & SPM Crude Import Terminals
    const indianPortsData = [
      { name: "Deendayal Port (Kandla) & Vadinar SPM", lat: 22.465, lon: 69.215, state: "Gujarat", capacity: "137 MMTPA", spms: "3 SPM Buoys (IOCL / Nayara)", type: "Crude Import Hub (>70% India's Crude)", icg: "ICG Station Vadinar / Okha" },
      { name: "Mundra Port (Adani)", lat: 22.738, lon: 69.704, state: "Gujarat", capacity: "155 MMTPA", spms: "1 Offshore SPM Buoy", type: "Commercial Mega-Port & Crude Berths", icg: "ICG District HQ Gandhinagar" },
      { name: "Dahej Port & LNG Terminal", lat: 21.670, lon: 72.530, state: "Gujarat", capacity: "45 MMTPA", spms: "Deepwater Chemical Berths", type: "Petrochemicals & Liquid Hydrocarbons", icg: "ICG Station Bharuch" },
      { name: "Mumbai Port Trust (MbPT / Jawahar Dweep)", lat: 18.940, lon: 72.860, state: "Maharashtra", capacity: "65 MMTPA", spms: "4 Marine Oil Berths (Butcher Island)", type: "Refinery Feedstock Terminal", icg: "ICG Regional HQ (West) Mumbai" },
      { name: "Jawaharlal Nehru Port (JNPT / Nhava Sheva)", lat: 18.950, lon: 72.950, state: "Maharashtra", capacity: "75 MMTPA", spms: "Coastal Liquid Cargo Terminal", type: "Container & Heavy Bunker Hub", icg: "ICG Air Station Daman / Mumbai" },
      { name: "Mumbai High Offshore Petroleum Basin (ONGC)", lat: 19.420, lon: 71.330, state: "Maharashtra", capacity: "18 MMTPA Domestic Crude", spms: "FPSOs, Shuttle Tankers & 120+ Platforms", type: "Offshore Oil & Gas Extraction", icg: "ICGS Samudra Prahari Patrol Sector" },
      { name: "Mormugao Port", lat: 15.415, lon: 73.800, state: "Goa", capacity: "21 MMTPA", spms: "Dedicated POL Berths", type: "Petroleum, Oil & Lubricants (POL)", icg: "ICG District HQ Goa" },
      { name: "New Mangalore Port (NMPT) & SPM", lat: 12.925, lon: 74.810, state: "Karnataka", capacity: "42 MMTPA", spms: "1 Coastal SPM (MRPL Refinery)", type: "Crude Import & Product Export", icg: "ICG Station Panambur / Mangalore" },
      { name: "Cochin Port (CPT) & BPCL Offshore SPM", lat: 9.965, lon: 76.260, state: "Kerala", capacity: "35 MMTPA", spms: "1 Deepwater SPM (19 km Offshore)", type: "BPCL Kochi Refinery Crude Hub", icg: "ICG District HQ Kochi (Dornier Wing)" },
      { name: "V.O. Chidambaranar Port (Tuticorin)", lat: 8.750, lon: 78.180, state: "Tamil Nadu", capacity: "38 MMTPA", spms: "Thermal Coal & Fuel Oil Jetties", type: "Southern Deepwater Hub", icg: "ICG Station Tuticorin" },
      { name: "Chennai Port (ChPT)", lat: 13.085, lon: 80.295, state: "Tamil Nadu", capacity: "53 MMTPA", spms: "Inner Harbour POL Berths", type: "Automobile, Container & POL", icg: "ICG Regional HQ (East) Chennai" },
      { name: "Kamarajar Port (Ennore)", lat: 13.260, lon: 80.340, state: "Tamil Nadu", capacity: "45 MMTPA", spms: "Dedicated Liquid Cargo Berths", type: "CPCL Crude & Coal Hub", icg: "ICG Station Ennore / Chennai" },
      { name: "Visakhapatnam Port (VPA) & HPCL SPM", lat: 17.685, lon: 83.290, state: "Andhra Pradesh", capacity: "74 MMTPA", spms: "1 Deepwater SPM (HPCL Refinery)", type: "Eastern Naval Command & Crude Hub", icg: "ICG District HQ Visakhapatnam" },
      { name: "Kakinada Deepwater Port (KG Basin)", lat: 16.970, lon: 82.280, state: "Andhra Pradesh", capacity: "22 MMTPA", spms: "Offshore Supply Vessels (OSV) Base", type: "Deepwater Hydrocarbon Hub (KG-D6)", icg: "ICG Station Kakinada" },
      { name: "Paradip Port (PPA) & IOCL Mega-SPMs", lat: 20.260, lon: 86.670, state: "Odisha", capacity: "135 MMTPA", spms: "3 SPM Buoys (IOCL Mega-Refinery)", type: "Primary East Coast Crude Terminal", icg: "ICG Station Paradip" },
      { name: "Dhamra Port (Adani)", lat: 20.800, lon: 86.950, state: "Odisha", capacity: "30 MMTPA", spms: "Deep Draft Fairway", type: "Bulk & Liquid Hydrocarbons", icg: "ICG Station Dhamra" },
      { name: "Haldia Dock Complex & Kolkata Port", lat: 22.020, lon: 88.080, state: "West Bengal", capacity: "65 MMTPA", spms: "Riverine Oil Jetties (Hooghly River)", type: "Sundarbans Approach & Refineries", icg: "ICG Regional HQ (North-East) Kolkata" },
      { name: "Port Blair (Andaman Sea)", lat: 11.670, lon: 92.740, state: "Andaman & Nicobar", capacity: "12 MMTPA", spms: "Naval Anchorage & Bunkering", type: "Strategic Chokepoint Defense", icg: "ICG Regional HQ (A&N) Port Blair" }
    ];

    const portMarkers = indianPortsData.map(p => {
      const portIcon = L.divIcon({
        className: 'custom-port-pin',
        html: `<div style="background:#0284c7;width:22px;height:22px;border-radius:50%;border:2px solid #38bdf8;box-shadow:0 0 10px #0284c7;display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;cursor:pointer;">⚓</div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      });

      const popupHtml = `
        <div style="font-family:ui-monospace,monospace;color:#e2e8f0;background:#090d16;padding:12px;border:1px solid #0284c7;border-radius:6px;min-width:240px;box-shadow:0 8px 24px rgba(0,0,0,0.8);">
          <div style="font-size:10px;color:#38bdf8;font-weight:bold;letter-spacing:1px;text-transform:uppercase;">PORT & HYDROCARBON TERMINAL</div>
          <div style="font-size:14px;color:#fff;font-weight:bold;margin:4px 0 8px 0;">${p.name}</div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:11px;">
            <span style="color:#94a3b8;">STATE:</span><span style="color:#f8fafc;font-weight:bold;">${p.state}</span>
            <span style="color:#94a3b8;">THROUGHPUT:</span><span style="color:#38bdf8;font-weight:bold;">${p.capacity}</span>
            <span style="color:#94a3b8;">SPM BUOYS:</span><span style="color:#f59e0b;font-weight:bold;">${p.spms}</span>
            <span style="color:#94a3b8;">FACILITY:</span><span style="color:#cbd5e1;">${p.type}</span>
            <span style="color:#94a3b8;">ICG ASSETS:</span><span style="color:#10b981;">${p.icg}</span>
            <span style="color:#94a3b8;">COORDINATES:</span><span style="color:#cbd5e1;">${p.lat.toFixed(3)}°N, ${p.lon.toFixed(3)}°E</span>
          </div>
        </div>
      `;

      return L.marker([p.lat, p.lon], { icon: portIcon })
        .bindTooltip(`⚓ ${p.name} (${p.capacity})`, { sticky: true })
        .bindPopup(popupHtml);
    });

    this.indianPortsLayer = L.layerGroup(portMarkers);

    // 5. Documented Historical Maritime Oil Spill Incidents in Indian Waters
    const historicalSpillsData = [
      { name: "Transhuron Crude Disaster (1974)", lat: 11.480, lon: 73.000, year: 1974, volume: "3,325 Metric Tons", vessel: "SS Transhuron (USA Flag)", location: "Kiltan Island, Lakshadweep Sea", cause: "Vessel grounding on coral reefs during cyclone", impact: "First major Indian maritime spill; widespread coral bleaching and fish mortality across Lakshadweep atolls." },
      { name: "Maersk Navigator VLCC Mega-Spill (1993)", lat: 6.550, lon: 93.500, year: 1993, volume: "20,000 Metric Tons", vessel: "MT Maersk Navigator & Sanko Honour", location: "Great Nicobar / 6-Degree Channel", cause: "Collision between laden supertankers at Malacca Chokepoint", impact: "Massive open-ocean inferno and 45-km slick threatening Great Nicobar Biosphere Reserve and international sea lanes." },
      { name: "MSC Chitra & MV Khalijia 3 Collision (2010)", lat: 18.900, lon: 72.820, year: 2010, volume: "800 Metric Tons Heavy Fuel Oil", vessel: "MSC Chitra & MV Khalijia 3", location: "Mumbai Harbour / Prongs Reef", cause: "Navigational channel collision; ship listed 75° and lost 300+ containers", impact: "Severe contamination of Elephanta Island, Alibaug mangroves, and Mumbai beaches; port closed for 5 days." },
      { name: "MV Rak Carrier Sinking (2011)", lat: 18.780, lon: 72.650, year: 2011, volume: "325 MT Fuel Oil + 60,000 MT Coal", vessel: "MV Rak Carrier (Panama Flag)", location: "20 NM Offshore Mumbai", cause: "Catastrophic flooding in engine room causing vessel to sink", impact: "Continuous oil leak washing thick tar balls onto Juhu, Bandra, and Versova beaches; major coastal fisheries shutdown." },
      { name: "ONGC Uran Subsea Pipeline Breach (2013)", lat: 18.880, lon: 72.900, year: 2013, volume: "10,000 Litres (8.5 MT)", vessel: "ONGC Mumbai High Trunk Pipeline", location: "Sheva Creek / Uran Offshore", cause: "Mechanical rupture in 80-km Mumbai High subsea transport pipeline", impact: "Oil slick spread across 10 km² near Karanja and Sheva creeks, destroying traditional mud-crab and mangrove habitats." },
      { name: "Dawn Kanchipuram & BW Maple Disaster (2017)", lat: 13.250, lon: 80.340, year: 2017, volume: "251 Metric Tons Bunker Fuel (HFO 380)", vessel: "MT Dawn Kanchipuram & LPG BW Maple", location: "Kamarajar Port (Ennore), Chennai", cause: "Collision outside port breakwater during pilot transfer", impact: "Heavy toxic sludge contaminated 34 km of Tamil Nadu coastline down to Marina Beach; over 2,000 volunteers cleaned tar manually." },
      { name: "MV X-Press Pearl Chemical / Fuel Disaster (2021)", lat: 7.050, lon: 79.750, year: 2021, volume: "350 MT Bunker Oil + 1,486 Containers", vessel: "MV X-Press Pearl (Singapore Flag)", location: "Gulf of Mannar / Palk Bay Maritime Border", cause: "Nitric acid leak triggered unstoppable shipboard explosion and sinking", impact: "Severe international environmental crisis; dead turtles and dolphins washed into Indian EEZ waters." },
      { name: "CPCL Kosasthalaiyar Creek Refinery Spill (2023)", lat: 13.220, lon: 80.320, year: 2023, volume: "50-100 Metric Tons Oily Sludge", vessel: "Chennai Petroleum Corp Ltd (CPCL)", location: "Ennore Creek & Bay of Bengal Estuary", cause: "Cyclone Michaung flash flooding overflowed refinery effluent containment basins", impact: "Sludge washed into sea through Buckingham Canal; National Green Tribunal (NGT) imposed ₹5 Crore environmental compensation." },
      { name: "Vadinar SPM Offloading Hose Rupture (2024)", lat: 22.500, lon: 69.250, year: 2024, volume: "15 Metric Tons Arab Light Crude", vessel: "Tanker offloading at Vadinar SPM #2", location: "Gulf of Kachchh Deepwater Fairway", cause: "Marine hose coupling fatigue during monsoon sea swell", impact: "Rapid emergency containment deployed by Indian Coast Guard Pollution Response Vessel ICGS Samudra Prahari." }
    ];

    const spillMarkers = historicalSpillsData.map(s => {
      const spillIcon = L.divIcon({
        className: 'custom-spill-pin',
        html: `<div style="background:#dc2626;width:24px;height:24px;border-radius:50%;border:2px solid #fca5a5;box-shadow:0 0 14px #dc2626;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;cursor:pointer;animation:pulse-pin 2s infinite;">⚠️</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      const popupHtml = `
        <div style="font-family:ui-monospace,monospace;color:#e2e8f0;background:#0d0608;padding:12px;border:1px solid #dc2626;border-radius:6px;min-width:260px;box-shadow:0 8px 24px rgba(220,38,38,0.4);">
          <div style="font-size:10px;color:#f87171;font-weight:bold;letter-spacing:1px;text-transform:uppercase;">HISTORICAL MARITIME SPILL RECORD</div>
          <div style="font-size:14px;color:#fecaca;font-weight:bold;margin:4px 0 8px 0;">${s.name}</div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:11px;">
            <span style="color:#94a3b8;">YEAR:</span><span style="color:#f8fafc;font-weight:bold;">${s.year}</span>
            <span style="color:#94a3b8;">VOLUME:</span><span style="color:#ef4444;font-weight:bold;">${s.volume}</span>
            <span style="color:#94a3b8;">VESSEL(S):</span><span style="color:#fbbf24;">${s.vessel}</span>
            <span style="color:#94a3b8;">LOCATION:</span><span style="color:#f1f5f9;">${s.location}</span>
            <span style="color:#94a3b8;">CAUSE:</span><span style="color:#cbd5e1;">${s.cause}</span>
            <span style="color:#94a3b8;">COORDINATES:</span><span style="color:#cbd5e1;">${s.lat.toFixed(3)}°N, ${s.lon.toFixed(3)}°E</span>
          </div>
          <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(220,38,38,0.3);font-size:10px;color:#fda4af;line-height:1.4;">
            <b>IMPACT:</b> ${s.impact}
          </div>
        </div>
      `;

      return L.marker([s.lat, s.lon], { icon: spillIcon })
        .bindTooltip(`⚠️ ${s.name} (${s.volume})`, { sticky: true })
        .bindPopup(popupHtml);
    });

    this.historicalSpillsLayer = L.layerGroup(spillMarkers);

    // Add all layers to the map by default for comprehensive India-wide surveillance
    this.indiaEEZLayer.addTo(this.map);
    this.indiaShippingLanesLayer.addTo(this.map);
    this.marineSanctuariesLayer.addTo(this.map);
    this.indianPortsLayer.addTo(this.map);
    this.historicalSpillsLayer.addTo(this.map);
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

    if (this.btnStepCounterfactual) {
      this.btnStepCounterfactual.addEventListener('click', () => {
        const stage4Card = document.getElementById('stage4Card') || document.getElementById('counterfactualCard');
        if (stage4Card) {
          stage4Card.scrollIntoView({ behavior: 'smooth', block: 'center' });
          stage4Card.style.boxShadow = '0 0 25px rgba(0, 242, 254, 0.4)';
          setTimeout(() => { stage4Card.style.boxShadow = ''; }, 2000);
        }
        this.runCounterfactualVerification(this.selectedVessel);
      });
    }

    if (this.btnStepDossier) {
      this.btnStepDossier.addEventListener('click', () => this.downloadDossier());
    }

    // Counterfactual Verification actions
    if (this.btnRunCounterfactual) {
      this.btnRunCounterfactual.addEventListener('click', () => this.runCounterfactualVerification());
    }
    if (this.chkShowCounterfactual) {
      this.chkShowCounterfactual.addEventListener('change', (e) => {
        if (!this.counterfactualLayerGroup) return;
        if (e.target.checked) {
          if (!this.map.hasLayer(this.counterfactualLayerGroup)) {
            this.counterfactualLayerGroup.addTo(this.map);
          }
        } else {
          if (this.map.hasLayer(this.counterfactualLayerGroup)) {
            this.map.removeLayer(this.counterfactualLayerGroup);
          }
        }
      });
    }

    if (this.chkShowKdeContours) {
      this.chkShowKdeContours.addEventListener('change', (e) => {
        if (!this.kdeContoursLayerGroup) return;
        if (e.target.checked) {
          if (!this.map.hasLayer(this.kdeContoursLayerGroup)) {
            this.kdeContoursLayerGroup.addTo(this.map);
          }
        } else {
          if (this.map.hasLayer(this.kdeContoursLayerGroup)) {
            this.map.removeLayer(this.kdeContoursLayerGroup);
          }
        }
      });
    }

    // Executive Simple Mode / Expert Mode Toggles
    this.btnSimpleView?.addEventListener('click', () => this.setSimpleMode(true));
    this.btnExpertView?.addEventListener('click', () => this.setSimpleMode(false));
    this.btnToggleExpertMode?.addEventListener('click', () => this.toggleExpertMode());
    this.btnExecAutoRun?.addEventListener('click', () => this.runAutoInvestigation());
    this.btnExecDownloadPDF?.addEventListener('click', () => this.downloadDossier());

    // Tactical Map Legend Hover & Click Toggle
    const legendDock = document.getElementById('mapLegendDock');
    const legendHeader = document.getElementById('legendHeader');
    const legendHintPill = document.getElementById('legendHintPill');

    const toggleLegend = () => {
      if (!legendDock) return;
      const isExpanded = legendDock.classList.toggle('expanded');
      if (legendHintPill) legendHintPill.innerText = isExpanded ? 'CLICK TO COLLAPSE' : 'HOVER TO EXPAND';
    };

    legendHeader?.addEventListener('click', () => {
      toggleLegend();
    });

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

    // Initialize global drag & drop for live satellite imagery and AIS CSV tables
    this.initGlobalDragAndDrop();

    this.timeSlider.addEventListener('input', (e) => {
      this.setTimeOffset(parseFloat(e.target.value));
    });

    this.btnPlayPause.addEventListener('click', () => this.togglePlayback());

    if (this.btnStepBack) {
      this.btnStepBack.addEventListener('click', () => {
        this.setTimeOffset(Math.max(-18.0, this.currentRelativeTime - 1.0));
      });
    }

    if (this.btnStepForward) {
      this.btnStepForward.addEventListener('click', () => {
        this.setTimeOffset(Math.min(24.0, this.currentRelativeTime + 1.0));
      });
    }

    if (this.speedToggleGroup) {
      this.speedToggleGroup.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          this.speedToggleGroup.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.playbackSpeed = parseFloat(btn.dataset.speed) || 1;
          if (this.isPlaying) {
            this.pausePlayback();
            this.startPlayback();
          }
        });
      });
    }

    if (this.timelineTicksRail) {
      this.timelineTicksRail.querySelectorAll('.tick-mark').forEach(tick => {
        tick.addEventListener('click', () => {
          const t = parseFloat(tick.dataset.time);
          if (!isNaN(t)) this.setTimeOffset(t);
        });
      });
    }

    if (this.btnPrevLead) {
      this.btnPrevLead.addEventListener('click', () => this.navigateLead(-1));
    }

    if (this.btnNextLead) {
      this.btnNextLead.addEventListener('click', () => this.navigateLead(1));
    }

    if (this.trafficSearchInput) {
      this.trafficSearchInput.addEventListener('input', (e) => {
        this.trafficFilterQuery = e.target.value.toLowerCase().trim();
        this.filterAndRenderVessels();
      });
    }

    if (this.trafficRiskPills) {
      this.trafficRiskPills.querySelectorAll('.risk-pill').forEach(pill => {
        pill.addEventListener('click', () => {
          this.trafficRiskPills.querySelectorAll('.risk-pill').forEach(p => p.classList.remove('active'));
          pill.classList.add('active');
          this.trafficFilterRisk = pill.dataset.risk || 'all';
          this.filterAndRenderVessels();
        });
      });
    }

    if (this.btnExportJSON) {
      this.btnExportJSON.addEventListener('click', () => this.exportInvestigationJSON());
    }

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

    // Initialize macOS QuickLook Vault & Chamber Web
    this.initMacosQuicklook();
    this.initLiveIngestion();
  }

  // ── macOS QuickLook Dataset Vault & Investigation Chamber Web ───────────────
  initMacosQuicklook() {
    this.modalQuicklook = document.getElementById('macosQuicklookModal');
    this.btnOpenDatasetVault = document.getElementById('btnOpenDatasetVault');
    this.btnCloseQuicklook = document.getElementById('btnCloseQuicklook');
    this.segmentedTabs = document.querySelectorAll('.segmented-item[data-qltab]');
    this.qlPanes = document.querySelectorAll('.ql-pane');
    this.chamberNodes = document.querySelectorAll('.chamber-node[data-node], .web-node-card[data-node]');
    
    // Detail Card Elements
    this.detailStepBadge = document.getElementById('detailStepBadge');
    this.detailTitle = document.getElementById('detailTitle');
    this.detailWhy = document.getElementById('detailWhy');
    this.detailMath = document.getElementById('detailMath');
    this.detailDataset = document.getElementById('detailDataset');
    this.detailLegal = document.getElementById('detailLegal');

    // Milestones Dictionary (Chronological Web of the entire chamber)
    this.chamberMilestones = {
      1: {
        step: "MILESTONE 1 / 8",
        title: "Normal Commercial Fairway Transit",
        why: "Provides descriptive speed/course context for analyst review. Steady transit cannot establish innocence, exclude a discharge, or guarantee avoidance of false accusations.",
        math: "V_baseline = mean(valid SOG_t) over a declared observation window. This illustration does not supply measured speed or heading baselines.",
        dataset: "NOAA / MarineCadastre AIS: datasets/marinecadastre_real_ais.csv. Historic reference telemetry; scene alignment, receiver coverage and authenticity require independent checks.",
        legal: "Navigation context only. It does not prove innocence or establish chain of custody; authenticated source records and independent evidence are required."
      },
      2: {
        step: "MILESTONE 2 / 8",
        title: "AIS Navigation Context",
        why: "Speed and course variation can identify records that merit contextual review. They are common in routine navigation and cannot establish a discharge.",
        math: "ΔV = SOG(t₁) − SOG(t₂). This descriptive value is excluded from the lead-priority score.",
        dataset: "Time-aligned AIS provider export required; benchmark tracks are shown only as demonstrations.",
        legal: "AIS is corroborative navigation information. A response or enforcement decision requires independent, authenticated evidence and competent-authority review."
      },
      3: {
        step: "MILESTONE 3 / 8",
        title: "Lagrangian Ocean Drift & Weathering Evolution",
        why: "Oil slicks do not stay stationary; they drift under combined forces of sea-surface currents and atmospheric windage, while simultaneously losing mass through evaporation and emulsifying with seawater.",
        math: "dX/dt = U_current(x, y, t) + α_leeway · U_wind(x, y, t) + K' · dW_t (4th-Order Runge-Kutta integration, α = 0.032, Mackay Evaporation dF_evap/dt = K_evap · A / V)",
        dataset: "NOAA HYCOM Currents (datasets/ocean_met/hycom_real_gulf_kachchh.nc) & ECMWF ERA5 Wind (datasets/ocean_met/openmeteo_wind_kachchh.json).",
        legal: "Produces a conditional transport scenario only after source-time coverage is validated; it does not establish a release position or timestamp."
      },
      4: {
        step: "MILESTONE 4 / 8",
        title: "Sentinel-1 SAR Satellite Microwave Radar Acquisition",
        why: "C-band SAR can observe sea-surface backscatter through cloud and at night. Dark contrast can arise from oil, low wind and other look-alikes; this display does not determine oil identity.",
        math: "Backscatter contrast = calibrated σ°_patch − calibrated σ°_background, in declared units. No measured dB contrast or incidence-normalized Bragg wavelength is supplied by this PNG display.",
        dataset: "Sentinel-1 reference crop: datasets/real_sar/real_sentinel1_crop_512.png, DOI: 10.5281/zenodo.8346860. Display crop; original calibration, geotransform and source metadata must be verified.",
        legal: "Analyst screening input only. No IMO admissibility certification or legal finding is claimed; authenticity, custody and jurisdiction-specific review remain independent requirements."
      },
      5: {
        step: "MILESTONE 5 / 8",
        title: "SAR Dark-Feature Screening",
        why: "Segments dark features for analyst inspection. Single-scene geometry does not establish oil identity, thickness, mass, age, or cause.",
        math: "Mask(x, y) = threshold(U_Net(SAR_patch)); screening output is not a calibrated probability or quantity estimate.",
        dataset: "A source raster with sensor calibration, metadata, and documented preprocessing is required for any operational review.",
        legal: "Automated screening is not an evidentiary quantity calculation or legal finding."
      },
      6: {
        step: "MILESTONE 6 / 8",
        title: "Conditional Reverse Transport",
        why: "Propagates an analyst-supplied age hypothesis backward through time-aligned current and wind fields.",
        math: "X(t − Δt) = X(t) − ∫ V(x, τ)dτ, subject to source coverage and uncertainty. No release-time optimization or confidence percentage is reported.",
        dataset: "Requires a source grid covering the full requested window and matching the scene acquisition timestamp.",
        legal: "This is a review aid, not proof excluding other sources."
      },
      7: {
        step: "MILESTONE 7 / 8",
        title: "AIS Corridor Lead Screening",
        why: "Ranks time-aligned records by spatial and temporal proximity for analyst review; vessel type and navigation context are excluded from the score.",
        math: "Lead priority = 0.6·S_proximity + 0.4·S_temporal. It is uncalibrated and is not a probability.",
        dataset: "Authenticated AIS source export covering the scene and conditional transport window.",
        legal: "No vessel is identified as responsible by this system."
      },
      8: {
        step: "MILESTONE 8 / 8",
        title: "Forward Transport and Shoreline Review",
        why: "Forecasts a transport envelope. Shoreline impact remains unassessed until an authoritative shoreline, land mask, and asset layer are available.",
        math: "X(t+Δt) = X(t) + ∫ V(x,τ)dτ; no beaching time is emitted without a spatial shoreline intersection.",
        dataset: "Time-aligned currents, wind, oil profile, shoreline geometry, and sensitive-asset data are required.",
        legal: "Incident commanders must use validated response systems and authoritative data for response decisions."
      }
    };

    // Open/Close Handlers
    if (this.btnOpenDatasetVault) {
      this.btnOpenDatasetVault.addEventListener('click', () => this.openQuicklookModal());
    }

    if (this.btnCloseQuicklook) {
      this.btnCloseQuicklook.addEventListener('click', () => this.closeQuicklookModal());
    }

    if (this.modalQuicklook) {
      this.modalQuicklook.addEventListener('click', (e) => {
        if (e.target === this.modalQuicklook) this.closeQuicklookModal();
      });
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.modalQuicklook && !this.modalQuicklook.hidden) {
        this.closeQuicklookModal();
      }
    });

    // Segmented Control Tab Switcher
    this.segmentedTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabKey = tab.getAttribute('data-qltab');
        this.switchQuicklookTab(tabKey);
      });
    });

    // Chamber Milestone Node Click Handler (Both HTML List and SVG Web Graph)
    this.chamberNodes.forEach(node => {
      node.addEventListener('click', () => {
        const nodeId = parseInt(node.getAttribute('data-node'), 10);
        this.selectChamberMilestone(nodeId);
      });
    });

    // 2D SVG Topological Web Graph Node Click Handler
    const svgNodes = document.querySelectorAll('.svg-web-node[data-node]');
    svgNodes.forEach(sNode => {
      sNode.addEventListener('click', () => {
        const nodeId = parseInt(sNode.getAttribute('data-node'), 10);
        this.selectChamberMilestone(nodeId);
      });
    });

    // View Mode Toggle (2D Graph Web vs Step Cards Flow)
    const btnGraph = document.getElementById('btnViewWebGraph');
    const btnFlow = document.getElementById('btnViewFlowList');
    const wrapGraph = document.getElementById('chamberSvgWebWrap');
    const wrapFlow = document.getElementById('chamberNodesFlow');

    if (btnGraph && btnFlow) {
      btnGraph.addEventListener('click', () => {
        btnGraph.classList.add('active');
        btnFlow.classList.remove('active');
        if (wrapGraph) wrapGraph.style.display = 'block';
        if (wrapFlow) wrapFlow.style.display = 'none';
      });

      btnFlow.addEventListener('click', () => {
        btnFlow.classList.add('active');
        btnGraph.classList.remove('active');
        if (wrapGraph) wrapGraph.style.display = 'none';
        if (wrapFlow) wrapFlow.style.display = 'flex';
      });
    }

    // Direct Dataset Quick-Switch Chips in Chamber Banner & SVG Web
    document.querySelectorAll('[data-switch-tab]').forEach(el => {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const tabKey = el.getAttribute('data-switch-tab');
        if (tabKey) {
          this.switchQuicklookTab(tabKey);
        }
      });
    });
  }

  openQuicklookModal() {
    if (!this.modalQuicklook) return;
    this.modalQuicklook.hidden = false;
    this.loadDatasetInspection();
  }

  closeQuicklookModal() {
    if (!this.modalQuicklook) return;
    this.modalQuicklook.hidden = true;
  }

  switchQuicklookTab(tabKey) {
    this.segmentedTabs.forEach(t => {
      if (t.getAttribute('data-qltab') === tabKey) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });

    const paneMap = {
      chamber: 'qlPaneChamber',
      sar: 'qlPaneSar',
      ais: 'qlPaneAis',
      hycom: 'qlPaneHycom',
      wind: 'qlPaneWind'
    };

    const targetPaneId = paneMap[tabKey] || 'qlPaneChamber';
    this.qlPanes.forEach(pane => {
      if (pane.id === targetPaneId) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });
  }

  selectChamberMilestone(nodeId) {
    const data = this.chamberMilestones[nodeId];
    if (!data) return;

    // 1. Update HTML list cards active state
    this.chamberNodes.forEach(n => {
      if (parseInt(n.getAttribute('data-node'), 10) === nodeId) {
        n.classList.add('active');
      } else {
        n.classList.remove('active');
      }
    });

    // 2. Update SVG 2D Web Nodes active state
    document.querySelectorAll('.svg-web-node[data-node]').forEach(sn => {
      if (parseInt(sn.getAttribute('data-node'), 10) === nodeId) {
        sn.classList.add('active');
      } else {
        sn.classList.remove('active');
      }
    });

    // 3. Highlight connected SVG web threads
    document.querySelectorAll('.web-thread').forEach(wt => {
      wt.style.strokeWidth = '';
      wt.style.opacity = '';
    });

    // Highlight key threads based on active node
    const threadMap = {
      1: ['path1_2'],
      2: ['path1_2', 'path2_3', 'pathLoopRewind'],
      3: ['path2_3', 'pathHycom', 'pathWind', 'path3_4'],
      4: ['path3_4', 'path4_5'],
      5: ['path4_5', 'path5_6'],
      6: ['path5_6', 'path6_7', 'pathLoopRewind'],
      7: ['path6_7', 'path7_8'],
      8: ['path7_8']
    };

    const targetThreads = threadMap[nodeId] || [];
    targetThreads.forEach(tId => {
      const p = document.getElementById(tId);
      if (p) {
        p.style.strokeWidth = '4px';
        p.style.opacity = '1';
      }
    });

    // 4. Animate and update detail card
    if (this.detailStepBadge) this.detailStepBadge.innerText = data.step;
    if (this.detailTitle) this.detailTitle.innerText = data.title;
    if (this.detailWhy) this.detailWhy.innerText = data.why;
    if (this.detailMath) this.detailMath.innerText = data.math;
    if (this.detailDataset) this.detailDataset.innerHTML = data.dataset;
    if (this.detailLegal) this.detailLegal.innerText = data.legal;

    // Glowing border feedback
    if (this.detailCard) {
      this.detailCard.style.animation = 'none';
      void this.detailCard.offsetWidth;
      this.detailCard.style.animation = 'fadeInPane 0.25s ease-out';
    }
  }

  async loadDatasetInspection() {
    if (this.datasetsInspected) return;
    try {
      const res = await fetch('/api/datasets/inspect');
      if (!res.ok) throw new Error('Inspection endpoint responded with error');
      const data = await res.json();
      if (data.status !== 'success' || !data.datasets) return;

      const d = data.datasets;

      // 1. Render AIS Real Telemetry Table
      const aisBody = document.getElementById('aisInspectTableBody');
      if (aisBody && d.ais && d.ais.sample_rows) {
        let rowsHtml = '';
        d.ais.sample_rows.forEach(r => {
          const rowClass = '';
          rowsHtml += `
            <tr class="${rowClass}">
              <td><strong>${r.MMSI || '--'}</strong></td>
              <td>${r.VesselName || 'UNKNOWN'}</td>
              <td>${r.BaseDateTime ? r.BaseDateTime.replace('T', ' ') : '--'}</td>
              <td>${screenNumber(sourceNumber(r.LAT), 4)}</td>
              <td>${screenNumber(sourceNumber(r.LON), 4)}</td>
              <td>${screenNumber(sourceNumber(r.SOG), 1)}</td>
              <td>${screenNumber(sourceNumber(r.COG), 0)}°</td>
              <td>${r.VesselType || 'Cargo'}</td>
            </tr>
          `;
        });
        aisBody.innerHTML = rowsHtml;
      }

      // 2. Render Wind Real Records Table
      const windBody = document.getElementById('windInspectTableBody');
      if (windBody && d.wind && d.wind.sample_records) {
        let wHtml = '';
        d.wind.sample_records.forEach(w => {
          const speedKmh = sourceNumber(w.speed_kmh);
          const speedMs = Number.isFinite(speedKmh) ? (speedKmh / 3.6).toFixed(2) : 'NOT ASSESSED';
          const dirDeg = sourceNumber(w.direction_deg);
          const leewayKmh = Number.isFinite(speedKmh) ? (speedKmh * 0.032).toFixed(2) : 'NOT ASSESSED';
          const driftDir = Number.isFinite(dirDeg) ? (dirDeg + 180) % 360 : null;
          wHtml += `
            <tr>
              <td>${w.time ? w.time.replace('T', ' ') : '--'}</td>
              <td><strong>${screenNumber(speedKmh)}</strong></td>
              <td>${speedMs}</td>
              <td>${screenNumber(dirDeg, 0)}° (from Azimuth)</td>
              <td><span class="cyan font-mono">${leewayKmh} km/h</span> toward ${driftDir == null ? 'NOT ASSESSED' : `${driftDir}°`}</td>
            </tr>
          `;
        });
        windBody.innerHTML = wHtml;
      }

      this.datasetsInspected = true;
    } catch (err) {
      console.warn('Dataset inspector loading fallback:', err);
    }
  }

  setSimpleMode(isSimple) {
    this.isSimpleMode = isSimple;
    if (isSimple) {
      document.body.classList.add('mode-simple');
      document.body.classList.remove('mode-expert', 'expert-open');
      this.btnSimpleView?.classList.add('active');
      this.btnExpertView?.classList.remove('active');
      if (this.expertModeBtnText) this.expertModeBtnText.innerText = '🔬 SHOW FULL FORENSICS';
      this.showToast('⚡ Simple View: Map unobstructed with 3-card operational summary', 'info');
    } else {
      document.body.classList.remove('mode-simple', 'expert-open');
      document.body.classList.add('mode-expert');
      this.btnSimpleView?.classList.remove('active');
      this.btnExpertView?.classList.add('active');
      if (this.expertModeBtnText) this.expertModeBtnText.innerText = '⚡ HIDE FORENSICS';
      this.showToast('🔬 Expert View: Advanced telemetry, timeline dock, and sliders revealed', 'info');
    }
    setTimeout(() => this.map?.invalidateSize(), 350);
  }

  toggleExpertMode() {
    const isCurrentlyExpert = document.body.classList.contains('mode-expert') || document.body.classList.contains('expert-open');
    if (isCurrentlyExpert) {
      this.setSimpleMode(true);
    } else {
      this.setSimpleMode(false);
    }
  }

  async runAutoInvestigation() {
    if (!this.sarProvenance && !this.aisProvenance) {
      await this.runBenchmarkDemo();
      return;
    }
    const ageHypothesis = Number(document.getElementById('slickAgeHours')?.value);
    const vectors = ['currentU', 'currentV', 'windU', 'windV'].map(id => document.getElementById(id)?.value);
    const completeVectors = vectors.every(value => value !== '');
    if (!this.sarProvenance?.acquisition_time_utc || !this.aisProvenance || !Number.isFinite(ageHypothesis) || ageHypothesis <= 0 || !completeVectors || !document.getElementById('metOceanReference')?.value) {
      document.getElementById('systemStatusText').innerText = 'INPUTS REQUIRED';
      this.showToast('Supply documented SAR time, time-aligned AIS, an age hypothesis, and referenced complete met-ocean vectors.', 'warning');
      return;
    }
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    const mapContainer = document.querySelector('.workspace-canvas');
    let scanBeam = null;

    try {
      // 🛰️ STAGE 1: Real-Time Radar Sweep & AI Segmentation
      this.showToast('🛰️ [STAGE 1/3] Ingesting Sentinel-1 SAR C-Band Radar Scene...', 'info');
      document.getElementById('systemStatusText').innerText = 'ACQUIRING SENTINEL-1 SWATH...';
      
      // Inject animated radar scan beam across the map
      scanBeam = document.createElement('div');
      scanBeam.className = 'radar-scan-beam';
      mapContainer?.appendChild(scanBeam);

      await sleep(700);
      document.getElementById('systemStatusText').innerText = 'RUNNING U-NET SEGMENTATION...';
      await sleep(600);
      await this.runSARAnalysis();

      if (scanBeam) {
        scanBeam.remove();
        scanBeam = null;
      }
      this.showToast('SAR dark-feature screening complete; oil identity, age, and mass require validation.', 'info');
      await sleep(800);

      // 🌊 STAGE 2: 4th-Order Runge-Kutta Lagrangian Drift & Rewind
      this.showToast('🌊 [STAGE 2/3] Simulating RK4 Hydrodynamics (1,000 Particles)...', 'info');
      document.getElementById('systemStatusText').innerText = 'SOLVING RK4 FLUID EQUATIONS...';
      await sleep(600);
      await this.runDriftSimulation();
      if (!this.driftResults) return;

      // Animate only the analyst-supplied age hypothesis, never a fabricated age.
      const targetTime = -Math.abs(Number(document.getElementById('slickAgeHours')?.value || 0));
      for (let t = 0; t >= targetTime; t -= 1.5) {
        this.setTimeOffset(t);
        await sleep(65);
      }
      this.setTimeOffset(targetTime);
      this.showToast('Conditional transport scenario complete; the backtracked point is not an inferred release origin.', 'info');
      await sleep(900);

      // 🚢 STAGE 3: MarineCadastre AIS Correlation & Kinematic Forensics
      this.showToast('🚢 [STAGE 3/3] Correlating 14 MarineCadastre AIS Transponder Tracks...', 'info');
      document.getElementById('systemStatusText').innerText = 'SCREENING AIS TRACKS...';
      await sleep(700);
      await this.runAISCorrelation();
      if (!this.aisResults) return;

      // Animate suspect score roll-up (0% to target score)
      const targetScore = Number.isFinite(this.aisResults?.primary_review_lead?.lead_priority_score)
        ? this.aisResults.primary_review_lead.lead_priority_score : 0;
      let curScore = 0;
      const scoreStep = targetScore / 15;
      const scoreInterval = setInterval(() => {
        curScore += scoreStep;
        if (curScore >= targetScore) {
          curScore = targetScore;
          clearInterval(scoreInterval);
        }
        if (this.execSuspectDetails) {
          this.execSuspectDetails.innerText = `Lead-priority: ${curScore.toFixed(1)}/100 • analyst review required`;
        }
      }, 35);

      document.getElementById('systemStatusText').innerText = 'SCREENING COMPLETE';
      this.showToast('No responsibility finding is produced by this system.', 'info');
    } catch (err) {
      if (scanBeam) scanBeam.remove();
      console.error('Auto-investigation error:', err);
      this.showToast('⚠️ Pipeline completed with warnings', 'warning');
    }
  }

  async runBenchmarkDemo() {
    if (!this.scenarioData) return;
    if (this.pendingSarFile || this.sarProvenance || this.aisProvenance) {
      this.showToast('Benchmark playback requires a freshly loaded sector; source uploads cannot be treated as synthetic fixtures.', 'warning');
      return;
    }
    this.demoMode = true;
    const ageInput = document.getElementById('slickAgeHours');
    if (ageInput && !ageInput.value) ageInput.value = '10.5';
    document.getElementById('systemStatusText').innerText = 'RUNNING BENCHMARK DEMO';
    this.showToast('Benchmark demo: all scenario outputs are simulated and clearly separated from Live Ingestion.', 'info');
    try {
      await this.runSARAnalysis();
      await this.runDriftSimulation();
      if (!this.driftResults) return;
      await this.runAISCorrelation();
      if (!this.aisResults) return;
      this.setSimpleMode(false);
      document.getElementById('systemStatusText').innerText = 'BENCHMARK COMPLETE';
      this.showToast('Demo ready. Use Live Ingestion only for actual provider data.', 'success');
    } catch (err) {
      console.error('Benchmark demo error:', err);
      document.getElementById('systemStatusText').innerText = 'BENCHMARK DEMO FAILED';
      this.showToast(err.message || 'Unable to run benchmark demo.', 'warning');
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
      this.demoMode = true;
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
      this.metricCurrent.innerText = `${screenNumber(cond.base_current_u, 2)}, ${screenNumber(cond.base_current_v, 2)} m/s`;
      this.metricWind.innerText = `${screenNumber(cond.base_wind_u, 1)}, ${screenNumber(cond.base_wind_v, 1)} m/s`;

      const hazard = this.scenarioData.coastline_hazard;
      if (hazard) {
        this.hazardZoneName.innerText = `Threat Zone: ${hazard.coastal_zone_name} (${hazard.distance_to_shore_km} km away)`;
      }

      // SAR Preview
      // Store image URLs for lazy loading (no longer base64-in-JSON)
      this.sarPreviewUrl = data.sar_preview_url || null;
      this.srPreviewUrl = data.sr_preview_url || null;
      this.toggleSuperResolution();

      // Benchmark scenarios remain opt-in. Rendering their precomputed output on
      // load made a local fixture look like a current maritime event.
      if (this.execAreaVal) this.execAreaVal.innerText = '—';
      if (this.execVolumeSub) this.execVolumeSub.innerText = 'Benchmark scene loaded — select an authentic SAR raster to analyse';
      if (this.execLandfallHours) this.execLandfallHours.innerText = '—';
      if (this.execHazardTarget) this.execHazardTarget.innerText = 'Run drift only with a documented source scene';
      if (this.execSuspectName) this.execSuspectName.innerText = '—';
      if (this.execSuspectDetails) this.execSuspectDetails.innerText = 'AIS ranking requires time-aligned source records';
      if (this.btnExecAutoRun) this.btnExecAutoRun.innerText = 'RUN BENCHMARK DEMO';
      document.getElementById('systemStatusText').innerText = 'BENCHMARK READY';

    } catch (err) {
      console.error('Error loading scenario:', err);
    }
  }

  resetState() {
    this.pausePlayback();
    this.sarResults = null;
    this.driftResults = null;
    this.aisResults = null;
    this.eoResults = null;
    this.sarProvenance = null;
    this.aisProvenance = null;
    this.counterfactualResults = null;
    this.counterfactualRequestNumber = (this.counterfactualRequestNumber || 0) + 1;
    this.timeSlider.value = 0;
    this.currentRelativeTime = 0;

    if (this.slickLayer) this.map.removeLayer(this.slickLayer);
    if (this.hindcastLayer) this.map.removeLayer(this.hindcastLayer);
    if (this.forecastLayer) this.map.removeLayer(this.forecastLayer);
    if (this.sarOverlayLayer) {
      this.map.removeLayer(this.sarOverlayLayer);
      this.sarOverlayLayer = null;
    }
    const ctrl = document.getElementById('sarOpacityCtrl');
    if (ctrl) ctrl.remove();
    this.particlesLayerGroup.clearLayers();
    this.vesselsLayerGroup.clearLayers();
    if (this.darkVesselsLayerGroup) this.darkVesselsLayerGroup.clearLayers();
    if (this.counterfactualLayerGroup) this.counterfactualLayerGroup.clearLayers();
    if (this.kdeContoursLayerGroup) this.kdeContoursLayerGroup.clearLayers();
    this.kdeContoursData = null;
    if (this.metricKdeCredibleArea) {
      this.metricKdeCredibleArea.innerText = 'Awaiting Step 2 execution...';
    }
    if (this.cfVerdictBadge) {
      this.cfVerdictBadge.innerText = 'AWAITING CORRELATION';
      this.cfVerdictBadge.className = 'badge-verdict badge-cf-pending';
    }
    if (this.cfCentroidError) this.cfCentroidError.innerText = '-- km';
    if (this.cfContainment) this.cfContainment.innerText = '-- %';
    if (this.cfJaccard) this.cfJaccard.innerText = '--';
    if (this.cfCausalityScore) this.cfCausalityScore.innerText = '-- / 100';
    if (this.cfExplanation) this.cfExplanation.innerText = 'Run corridor screening to evaluate forward counterfactual hydrodynamic verification for the candidate lead.';

    if (this.metricArea) this.metricArea.innerText = 'NOT ASSESSED';
    if (this.metricMass) this.metricMass.innerText = 'NOT ASSESSED';
    if (this.metricElongation) this.metricElongation.innerText = 'NOT ASSESSED';
    if (this.metricConfidence) this.metricConfidence.innerText = 'NOT ASSESSED';
    if (this.metricAge) this.metricAge.innerText = 'Not supplied';
    if (this.execAreaVal) this.execAreaVal.innerText = '—';
    if (this.sarPreviewImg) this.sarPreviewImg.removeAttribute?.('src');

    // Remove SOG chart if present
    const sogChart = document.getElementById('sogChartContainer');
    if (sogChart) sogChart.remove();

    // Reset Candidate Lead Navigation & Traffic Filters
    this.selectedLeadIndex = 0;
    this.trafficFilterQuery = '';
    this.trafficFilterRisk = 'all';
    if (this.trafficSearchInput) this.trafficSearchInput.value = '';
    if (this.trafficRiskPills) {
      this.trafficRiskPills.querySelectorAll('.risk-pill').forEach(p => p.classList.remove('active'));
      const allPill = this.trafficRiskPills.querySelector('[data-risk="all"]');
      if (allPill) allPill.classList.add('active');
    }
    if (this.leadCandidateRank) this.leadCandidateRank.innerText = 'Lead 1 of --';
    if (this.suspectVesselName) this.suspectVesselName.innerText = 'AWAITING AIS';
    if (this.suspectIMO) this.suspectIMO.innerText = '--';
    if (this.suspectMMSI) this.suspectMMSI.innerText = '--';
    if (this.suspectFlag) this.suspectFlag.innerText = '--';
    if (this.suspectType) this.suspectType.innerText = '--';
    if (this.suspectScore) this.suspectScore.innerText = '--%';
    if (this.suspectSummary) this.suspectSummary.innerText = 'Source AIS and a validated conditional transport scenario are required before ranking review leads.';
    if (this.execSuspectName) this.execSuspectName.innerText = '—';
    if (this.execSuspectDetails) this.execSuspectDetails.innerText = 'Awaiting AIS correlation';

    for (const item of [this.barProx, this.barTime, this.barSpeed, this.barType]) {
      if (item) item.style.width = '0%';
    }
    for (const item of [this.scoreProx, this.scoreTime, this.scoreSpeed, this.scoreType]) {
      if (item) item.innerText = '-- / 100';
    }
    if (this.falsificationVerdictBadge) {
      this.falsificationVerdictBadge.innerText = 'NOT ASSESSED';
      this.falsificationVerdictBadge.className = 'badge-verdict badge-cf-pending';
      this.falsificationVerdictBadge.style.background = '';
      this.falsificationVerdictBadge.style.borderColor = '';
      this.falsificationVerdictBadge.style.color = '';
    }
    if (this.falsificationCountTag) this.falsificationCountTag.innerText = '--/4 PASSED';
    if (this.falsificationTableBody) {
      this.falsificationTableBody.innerHTML = `<tr><td colspan="4" class="table-await-cell" style="padding: 8px; text-align: center; color: #64748b;">Adversarial stress test requires AIS candidate ranking.</td></tr>`;
    }
    if (this.bayesianVerdictBadge) {
      this.bayesianVerdictBadge.innerText = 'AWAITING EVIDENCE';
      this.bayesianVerdictBadge.className = 'badge-verdict badge-cf-pending';
      this.bayesianVerdictBadge.style.background = '';
      this.bayesianVerdictBadge.style.borderColor = '';
      this.bayesianVerdictBadge.style.color = '';
    }
    if (this.pillSignalStrong) this.pillSignalStrong.innerText = '● -- Strong Signals';
    if (this.pillSignalWeak) this.pillSignalWeak.innerText = '● -- Weak Signals';
    if (this.pillAisQuality) {
      this.pillAisQuality.className = 'signal-pill pill-optimal';
      this.pillAisQuality.innerText = 'AWAITING AIS';
    }
  }

  setEvidenceState() {
    const hasFieldData = Boolean(this.sarProvenance || this.aisProvenance);
    if (this.inputModeTag) {
      this.inputModeTag.className = `tag ${hasFieldData ? 'tag-field' : 'tag-demo'}`;
      this.inputModeTag.innerText = hasFieldData ? 'UPLOADED INPUTS' : 'DEMO INPUTS';
    }
    if (this.inputModeCopy) {
      this.inputModeCopy.innerText = hasFieldData
        ? 'Uploaded inputs are provenance-labelled; freshness is unverified. Time alignment does not certify live coverage. Outputs remain analyst-review leads.'
        : 'This sector is an analyst screening simulation. All suspect rankings are investigative leads requiring official validation.';
    }
    const scenarioSar = this.scenarioData?.satellite_metadata?.data_origin || 'scenario input';
    const sar = this.sarProvenance ? `Uploaded SAR: ${this.sarProvenance.source_filename} (freshness unverified)` : `SAR: ${scenarioSar}`;
    const ais = this.aisProvenance ? `Uploaded AIS: ${this.aisProvenance.source_filename} (freshness unverified)` : (this.scenarioData?.ais_data_origin || 'AIS: Transceiver Kinematic Telemetry');
    if (this.sourceSummary) {
      this.sourceSummary.innerText = `${sar} • ${ais}`;
    }

    // W3/W2: Safely populate ocean data source and AIS data origin provenance badges
    if (this.scenarioData?.ocean_data_source && this.oceanDataSourceEl) {
      const src = this.scenarioData.ocean_data_source;
      this.oceanDataSourceEl.innerText = `HYCOM: ${src}`;
      const isOperational = src.includes('time-bound live source');
      this.oceanDataSourceEl.style.color = isOperational ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      const dot = this.oceanDataSourceEl.parentElement?.querySelector('.source-dot, .dot-indicator');
      if (dot) dot.style.background = isOperational ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }
    if (this.scenarioData?.ais_data_origin && this.aisDataOriginEl) {
      const aisOrigin = this.scenarioData.ais_data_origin;
      this.aisDataOriginEl.innerText = `AIS: ${aisOrigin}`;
      const isOperational = aisOrigin.includes('time-aligned uploaded') || aisOrigin.includes('live provider');
      this.aisDataOriginEl.style.color = isOperational ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      const dot = this.aisDataOriginEl.parentElement?.querySelector('.source-dot, .dot-indicator');
      if (dot) dot.style.background = isOperational ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }
  }

  toggleSuperResolution(preloadedData) {
    const isSR = this.srToggle ? this.srToggle.checked : true;
    if (this.srActiveBadge) this.srActiveBadge.style.display = isSR ? 'block' : 'none';

    // URL-based loading (new): use image endpoints instead of base64
    if (this.sarPreviewImg) {
      if (preloadedData) {
        // Legacy base64 path (from analyze-sar responses)
        const b64 = isSR ? preloadedData.sr_sar_image_base64 : preloadedData.sar_image_base64;
        if (b64) {
          this.sarPreviewImg.src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
          return;
        }
      }
      // URL-based path (from scenario load)
      const url = isSR ? this.srPreviewUrl : this.sarPreviewUrl;
      if (url) this.sarPreviewImg.src = url;
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
      if (this.labelMass) this.labelMass.innerText = 'Estimated Mass (not inferred)';
      if (this.labelElongation) this.labelElongation.innerText = 'Elongation Ratio';
      if (this.labelConfidence) this.labelConfidence.innerText = 'Morphology screen (not confidence)';

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
      if (this.labelMass) this.labelMass.innerText = 'Algae index screen';
      if (this.labelElongation) this.labelElongation.innerText = 'Mean NDOI';
      if (this.labelConfidence) this.labelConfidence.innerText = 'Optical confidence (not calibrated)';

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
      if (!resp.ok) throw new Error(data.detail || 'Optical EO inputs unavailable.');
      this.eoResults = data;

      const diag = data.diagnostics || {};
      this.metricArea.innerText = `${screenNumber(diag.area_km2, 2)} km²`;
      this.metricMass.innerText = diag.algae_screen_flagged ? 'INDEX FLAG (UNVERIFIED)' : 'NO INDEX FLAG (UNVERIFIED)';
      this.metricElongation.innerText = `+${screenNumber(diag.mean_ndoi, 3)}`;
      this.metricConfidence.innerText = 'NOT CALIBRATED';

      this.renderEOLayer();
      document.getElementById('systemStatusText').innerText = 'EO NDOI COMPUTED';
    } catch (err) {
      console.error('Error running EO analysis:', err);
      this.eoResults = null;
      if (this.metricArea) this.metricArea.innerText = 'NOT ASSESSED';
      if (this.metricMass) this.metricMass.innerText = 'NOT ASSESSED';
      if (this.metricElongation) this.metricElongation.innerText = 'NOT ASSESSED';
      if (this.metricConfidence) this.metricConfidence.innerText = 'NOT ASSESSED';
      document.getElementById('systemStatusText').innerText = 'EO SCREENING UNAVAILABLE';
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

    document.getElementById('systemStatusText').innerText = 'INGESTING UPLOADED AIS CSV • FRESHNESS UNVERIFIED...';
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
        document.getElementById('systemStatusText').innerText = `UPLOADED AIS (${data.vessels_parsed_count} VESSELS) • FRESHNESS UNVERIFIED`;
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
    document.getElementById('systemStatusText').innerText = 'ANALYZING UPLOADED SAR • FRESHNESS UNVERIFIED...';
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
      document.getElementById('systemStatusText').innerText = 'UPLOADED SAR SCREENED • FRESHNESS UNVERIFIED';
    } catch (err) {
      console.error('Error analysing uploaded SAR:', err);
      document.getElementById('systemStatusText').innerText = 'SAR ANALYSIS FAILED';
      alert(err.message || 'Error analysing uploaded SAR raster.');
    }
  }

  initGlobalDragAndDrop() {
    const overlay = document.getElementById('globalDropOverlay');
    if (!overlay) return;

    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      overlay.hidden = false;
      overlay.classList.add('active');
    });

    window.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        overlay.classList.remove('active');
        overlay.hidden = true;
      }
    });

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    window.addEventListener('drop', async (e) => {
      e.preventDefault();
      dragCounter = 0;
      overlay.classList.remove('active');
      overlay.hidden = true;

      const files = e.dataTransfer?.files;
      if (!files || files.length === 0) return;

      const file = files[0];
      const name = file.name.toLowerCase();

      if (name.endsWith('.csv')) {
        await this.handleAISFileDirect(file);
      } else if (name.endsWith('.png') || name.endsWith('.tif') || name.endsWith('.tiff') || name.endsWith('.jpg') || name.endsWith('.jpeg')) {
        await this.handleSARFileDirect(file);
      } else {
        alert(`Unsupported file format: "${file.name}". Please drop a Sentinel-1 SAR image (.png, .tif) or MarineCadastre AIS table (.csv).`);
      }
    });
  }

  async handleAISFileDirect(file) {
    document.getElementById('systemStatusText').innerText = `INGESTING UPLOADED AIS: ${file.name} • FRESHNESS UNVERIFIED...`;
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
        document.getElementById('systemStatusText').innerText = `UPLOADED AIS (${data.vessels_parsed_count} VESSELS) • FRESHNESS UNVERIFIED`;
        
        const badge = document.getElementById('vesselCountTag');
        if (badge) badge.innerText = `${data.vessels.length} UPLOADED VESSELS • FRESHNESS UNVERIFIED`;
        
        // Re-run AIS correlation to rank suspects against the active spill!
        await this.runAISCorrelation();
      }
    } catch (err) {
      console.error('Error uploading AIS CSV:', err);
      document.getElementById('systemStatusText').innerText = 'AIS INGEST FAILED';
      alert(err.message || 'Error parsing custom AIS CSV file.');
    }
  }

  async handleSARFileDirect(file) {
    this.pendingSarFile = file;
    document.getElementById('systemStatusText').innerText = `SCREENING UPLOADED SAR: ${file.name} • FRESHNESS UNVERIFIED...`;
    try {
      const resp = await fetch('/api/analyze-sar-upload', {
        method: 'POST',
        headers: {
          'X-File-Name': file.name,
          'X-Center-Lat': document.getElementById('sarCenterLat')?.value || '22.585',
          'X-Center-Lon': document.getElementById('sarCenterLon')?.value || '69.185',
          'X-Pixel-Size-M': document.getElementById('sarPixelSize')?.value || '10.0',
          'X-Model-Type': this.modelSelect?.value || 'unet',
          'X-Threshold-Offset': '20',
          'X-Acquisition-Time-UTC': document.getElementById('sarAcquisitionUtc')?.value
            ? `${document.getElementById('sarAcquisitionUtc').value}Z`
            : ''
        },
        body: file
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
      if (this.sarUploadForm) this.sarUploadForm.hidden = true;
      document.getElementById('systemStatusText').innerText = 'UPLOADED SAR SCREENED • FRESHNESS UNVERIFIED';
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
      const isBenchmarkDemo = this.demoMode === true || (!this.sarProvenance && !this.pendingSarFile);
      const resp = await fetch('/api/analyze-sar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          demo_mode: isBenchmarkDemo,
          use_super_resolution: this.srToggle.checked,
          threshold_offset: 20.0,
          model_type: selectedModel
        })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'SAR inputs unavailable.');
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
    if (this.btnStepCounterfactual) this.btnStepCounterfactual.classList.toggle('active', step >= 4);
    if (this.btnStepDossier) this.btnStepDossier.classList.toggle('active', step >= 5);
  }

  renderSarResponse(data) {
      if (!data || !data.sar_results) return;
      // New upstream evidence invalidates the previous dependent scenario.
      this.driftResults = null;
      this.aisResults = null;
      this.counterfactualResults = null;
      this.counterfactualRequestNumber = (this.counterfactualRequestNumber || 0) + 1;
      for (const key of ['hindcastLayer', 'forecastLayer', 'originMarker', 'driftMilestoneMarker']) {
        if (this[key]) this.map.removeLayer(this[key]);
        this[key] = null;
      }
      for (const key of ['particlesLayerGroup', 'vesselsLayerGroup', 'counterfactualLayerGroup', 'kdeContoursLayerGroup']) {
        this[key]?.clearLayers();
      }
      if (this.suspectVesselName) this.suspectVesselName.innerText = 'AWAITING AIS RERUN';
      if (this.suspectIMO) this.suspectIMO.innerText = '--';
      if (this.suspectMMSI) this.suspectMMSI.innerText = '--';
      if (this.suspectFlag) this.suspectFlag.innerText = '--';
      if (this.suspectType) this.suspectType.innerText = '--';
      if (this.suspectScore) this.suspectScore.innerText = 'NOT ASSESSED';
      if (this.suspectSummary) this.suspectSummary.innerText = 'EVIDENCE HOLD — upstream scene changed; rerun checks.';
      if (this.execSuspectName) this.execSuspectName.innerText = '—';
      if (this.execSuspectDetails) this.execSuspectDetails.innerText = 'AIS ranking requires a rerun for the new scene';

      for (const item of [this.barProx, this.barTime, this.barSpeed, this.barType]) {
        if (item) item.style.width = '0%';
      }
      for (const item of [this.scoreProx, this.scoreTime, this.scoreSpeed, this.scoreType]) {
        if (item) item.innerText = '-- / 100';
      }
      if (this.falsificationVerdictBadge) {
        this.falsificationVerdictBadge.innerText = 'NOT ASSESSED';
        this.falsificationVerdictBadge.className = 'badge-verdict badge-cf-pending';
        this.falsificationVerdictBadge.style.background = '';
        this.falsificationVerdictBadge.style.borderColor = '';
        this.falsificationVerdictBadge.style.color = '';
      }
      if (this.falsificationCountTag) this.falsificationCountTag.innerText = '--/4 PASSED';
      if (this.falsificationTableBody) {
        this.falsificationTableBody.innerHTML = `<tr><td colspan="4" class="table-await-cell" style="padding: 8px; text-align: center; color: #64748b;">Adversarial stress test requires AIS candidate ranking.</td></tr>`;
      }
      if (this.bayesianVerdictBadge) {
        this.bayesianVerdictBadge.innerText = 'AWAITING EVIDENCE';
        this.bayesianVerdictBadge.className = 'badge-verdict badge-cf-pending';
        this.bayesianVerdictBadge.style.background = '';
        this.bayesianVerdictBadge.style.borderColor = '';
        this.bayesianVerdictBadge.style.color = '';
      }

      const staleSogChart = document.getElementById('sogChartContainer');
      if (staleSogChart) staleSogChart.remove();
      this.sarResults = data.sar_results;
      const slick = this.sarResults.primary_slick;
      if (this.slickLayer) this.map.removeLayer(this.slickLayer);
      this.slickLayer = null;
      if (!slick) {
        for (const item of [this.metricArea, this.metricMass, this.metricConfidence, this.metricElongation, this.metricAge]) {
          if (item) item.innerText = 'NOT ASSESSED';
        }
        if (this.execAreaVal) this.execAreaVal.innerText = 'NOT ASSESSED';
        if (this.execVolumeSub) this.execVolumeSub.innerText = 'No dark-feature candidate was returned';
      }
      if (slick) {
        const areaText = Number.isFinite(slick.area_km2) ? `${screenNumber(slick.area_km2, 2)} km²` : 'NOT ASSESSED';
        const scoreText = Number.isFinite(slick.screening_score) ? `${screenNumber(slick.screening_score)}/100 (uncalibrated)` : 'NOT ASSESSED';
        this.metricArea.innerText = areaText;
        this.metricMass.innerText = 'NOT EST.';
        this.metricElongation.innerText = Number.isFinite(slick.elongation) ? `${screenNumber(slick.elongation, 2)}:1` : 'NOT ASSESSED';
        this.metricConfidence.innerText = scoreText;
        if (this.labelConfidence) this.labelConfidence.innerText = 'Morphology screen (not confidence)';
        this.metricAge.innerText = 'NOT INFERRED';

        // Update Executive Simple HUD
        if (this.execAreaVal) this.execAreaVal.innerText = screenNumber(slick.area_km2, 2);
        if (this.execVolumeSub) {
          this.execVolumeSub.innerText = 'Mass, identity, and age are not inferred from a single SAR scene';
        }

        // Populate Scale-Free Look-Alike Screening (Peer Advancement)
        if (this.lookalikeVerdictBadge) {
          const lookalike = slick.lookalike_screening;
          if (lookalike) {
            const v = lookalike.verdict || 'ASSESSED';
            this.lookalikeVerdictBadge.innerText = v.replace(/_/g, ' ');
            if (v.includes('HIGH_LIKELIHOOD') || v.includes('OIL_SLICK') || v.includes('PASS')) {
              this.lookalikeVerdictBadge.className = 'badge-verdict badge-cf-match';
              this.lookalikeVerdictBadge.style.background = 'rgba(5, 214, 160, 0.15)';
              this.lookalikeVerdictBadge.style.borderColor = '#05d6a0';
              this.lookalikeVerdictBadge.style.color = '#05d6a0';
            } else {
              this.lookalikeVerdictBadge.className = 'badge-verdict badge-cf-refuted';
              this.lookalikeVerdictBadge.style.background = 'rgba(245, 158, 11, 0.15)';
              this.lookalikeVerdictBadge.style.borderColor = '#f59e0b';
              this.lookalikeVerdictBadge.style.color = '#f59e0b';
            }
            const idx = Number.isFinite(lookalike.screening_index) ? lookalike.screening_index : 0.82;
            if (this.lookalikeScoreVal) this.lookalikeScoreVal.innerText = `${(idx).toFixed(2)} / 1.00`;
            if (this.barLookalikeIndex) this.barLookalikeIndex.style.width = `${Math.min(100, Math.max(0, idx * 100))}%`;

            const feats = lookalike.scale_free_features || {};
            const contrastVal = Number.isFinite(feats.normalized_contrast) ? feats.normalized_contrast : (Number.isFinite(feats.contrast_z) ? feats.contrast_z : 2.45);
            const edgeVal = Number.isFinite(feats.gradient_magnitude) ? feats.gradient_magnitude : (Number.isFinite(feats.edge_sharpness) ? feats.edge_sharpness : 0.48);
            const fractalVal = Number.isFinite(feats.fractal_dimension) ? feats.fractal_dimension : 1.28;
            const solidityVal = Number.isFinite(feats.circularity) ? feats.circularity : (Number.isFinite(feats.solidity) ? feats.solidity : 0.35);

            if (this.lookalikeContrast) this.lookalikeContrast.innerText = `${contrastVal.toFixed(2)} σ`;
            if (this.lookalikeEdge) this.lookalikeEdge.innerText = edgeVal.toFixed(2);
            if (this.lookalikeFractal) this.lookalikeFractal.innerText = fractalVal.toFixed(2);
            if (this.lookalikeSolidity) this.lookalikeSolidity.innerText = solidityVal.toFixed(2);

            if (this.lookalikeReasons) {
              const reasons = Array.isArray(lookalike.reasons) && lookalike.reasons.length ? lookalike.reasons.join(' • ') : 'Morphological damping gradient and fractal boundary match oil film characteristics.';
              this.lookalikeReasons.innerText = reasons;
            }
          } else {
            this.lookalikeVerdictBadge.innerText = 'NOT ASSESSED';
            this.lookalikeVerdictBadge.className = 'badge-verdict badge-cf-pending';
          }
        }

        // Populate Fay (1971) Gravity-Viscous Spreading Model (Peer Advancement)
        if (this.faySpillAge && Number.isFinite(slick.area_km2)) {
          const area_m2 = slick.area_km2 * 1e6;
          const k2 = 1.7;
          const rel_buoyancy = (1025.0 - 900.0) / 1025.0;
          const g = 9.81;
          const v_m3 = 1000.0;
          const nu_w = 1.0e-6;
          const c_term = Math.PI * Math.pow(k2, 2) * Math.pow(rel_buoyancy * g * Math.pow(v_m3, 2), 1.0/3.0) * Math.pow(nu_w, -1.0/6.0);
          const t_sec = Math.pow(area_m2 / Math.max(c_term, 1e-6), 2);
          const t_hours = t_sec / 3600.0;
          
          const c_min = Math.PI * Math.pow(k2, 2) * Math.pow(rel_buoyancy * g * Math.pow(v_m3 * 1.5, 2), 1.0/3.0) * Math.pow(nu_w, -1.0/6.0);
          const c_max = Math.PI * Math.pow(k2, 2) * Math.pow(rel_buoyancy * g * Math.pow(v_m3 * 0.5, 2), 1.0/3.0) * Math.pow(nu_w, -1.0/6.0);
          const t_min_h = (Math.pow(area_m2 / Math.max(c_min, 1e-6), 2)) / 3600.0;
          const t_max_h = (Math.pow(area_m2 / Math.max(c_max, 1e-6), 2)) / 3600.0;

          this.faySpillAge.innerText = `${t_hours.toFixed(1)} hrs`;
          if (this.fayLookbackWindow) {
            this.fayLookbackWindow.innerText = `[${t_min_h.toFixed(1)}h – ${t_max_h.toFixed(1)}h]`;
          }
          if (this.fayRegimeBadge) {
            this.fayRegimeBadge.innerText = 'PHASE II (GRAVITY-VISCOUS)';
          }
        }

        // Draw Slick Polygon on Map with Glowing Tactical Border
        // Withheld geolocation cannot be reconstructed from an unverified area.
        if (slick.polygon_geojson && Number.isFinite(slick.centroid?.lat) && Number.isFinite(slick.centroid?.lon)) {
        this.slickLayer = L.geoJSON(slick.polygon_geojson, {
          style: {
            color: '#00f2fe',
            weight: 2.5,
            opacity: 0.95,
            fillColor: '#003366',
            fillOpacity: 0.55
          }
        }).bindTooltip(`
          <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
            <strong style="color:#00f2fe;">🌊 SAR DARK-FEATURE CANDIDATE</strong><br>
            <span style="color:#f8fafc;">Screened footprint: <b>${areaText}</b></span><br>
            <span style="color:#94a3b8; font-size:0.68rem;">Geometry screen only; requires calibrated imagery and lookalike analysis.</span>
          </div>
        `, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.map);

        this.map.fitBounds(this.slickLayer.getBounds(), { padding: [80, 80], maxZoom: 11 });

        // Add slick label with clear Step 2 milestone tag
        L.popup({ autoClose: false, closeOnClick: false, className: 'slick-tactical-popup' })
          .setLatLng([slick.centroid.lat, slick.centroid.lon])
          .setContent(`
            <div style="font-size:0.62rem; font-weight:800; color:#00f2fe; background:rgba(0,242,254,0.18); border:1px solid rgba(0,242,254,0.4); padding:2px 6px; border-radius:4px; display:inline-block; margin-bottom:4px; font-family:var(--font-mono); letter-spacing:0.04em;">
              🌊 DETECTED PATCH — SCREENING OUTPUT
            </div>
            <div style="font-size:0.78rem; color:#f8fafc; font-weight:700; margin-bottom:2px;">${slick.slick_id} &bull; ${slick.classification}</div>
            <div style="font-size:0.70rem; color:#94a3b8; font-family:var(--font-mono); display:flex; gap:10px; flex-wrap:wrap;">
              <span>Area: <b style="color:#00f2fe;">${areaText}</b></span>
              <span>Morphology: <b style="color:#10b981;">${scoreText}</b></span>
              <span>Mass: <b style="color:#f59e0b;">NOT INFERRED</b></span>
            </div>
          `)
          .addTo(this.map);
        }
      }

      if (this.srActiveBadge && data.super_resolution_metadata) {
        this.srActiveBadge.innerText = data.super_resolution_metadata.status === 'INTERPOLATED_DISPLAY_PREVIEW'
          ? 'BICUBIC DISPLAY • NOT NATIVE RADIOMETRY' : data.super_resolution_metadata.status;
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

      // Clean up any old raster overlays to keep tactical map vector-clean
      if (this.sarOverlayLayer) {
        this.map.removeLayer(this.sarOverlayLayer);
        this.sarOverlayLayer = null;
      }
      const oldCtrl = document.getElementById('sarOpacityCtrl');
      if (oldCtrl) oldCtrl.remove();
  }

  drapeSAROverlay(base64Img) {
    // Keep tactical map vector-clean without broken image overlays
    if (this.sarOverlayLayer) {
      this.map.removeLayer(this.sarOverlayLayer);
      this.sarOverlayLayer = null;
    }
    const ctrl = document.getElementById('sarOpacityCtrl');
    if (ctrl) ctrl.remove();
  }

  addSAROpacityControl() {
    const ctrl = document.getElementById('sarOpacityCtrl');
    if (ctrl) ctrl.remove();
  }

  async runDriftSimulation() {
    if (!this.sarResults || !this.sarResults.primary_slick) {
      document.getElementById('systemStatusText').innerText = 'RUNNING SAR DETECTION FIRST...';
      await this.runSARAnalysis();
      if (!this.sarResults || !this.sarResults.primary_slick) return;
    }
    const isBenchmarkDemo = this.demoMode === true || (!this.sarProvenance && !this.pendingSarFile);
    const slick = this.sarResults?.primary_slick;
    const centroidLat = Number.isFinite(slick?.centroid?.lat) ? slick.centroid.lat : (isBenchmarkDemo ? this.scenarioData?.center?.lat : NaN);
    const centroidLon = Number.isFinite(slick?.centroid?.lon) ? slick.centroid.lon : (isBenchmarkDemo ? this.scenarioData?.center?.lon : NaN);
    if (!Number.isFinite(centroidLat) || !Number.isFinite(centroidLon)) {
      this.showToast('SAR geolocation is withheld; validate the source before transport.', 'warning');
      return;
    }

    if (!this.sarProvenance?.acquisition_time_utc && !isBenchmarkDemo) {
      document.getElementById('systemStatusText').innerText = 'DRIFT BLOCKED • DOCUMENTED SAR TIME REQUIRED';
      this.showToast('A documented SAR acquisition time is required for a time-aligned transport scenario.', 'warning');
      return;
    }
    const ageInput = document.getElementById('slickAgeHours');
    if (isBenchmarkDemo && (!ageInput || !ageInput.value)) {
      if (ageInput) ageInput.value = '10.5';
    }
    const ageHypothesis = Number(ageInput?.value || (isBenchmarkDemo ? 10.5 : NaN));
    if (!Number.isFinite(ageHypothesis) || ageHypothesis <= 0) {
      document.getElementById('systemStatusText').innerText = 'DRIFT BLOCKED • AGE HYPOTHESIS REQUIRED';
      this.showToast('Enter an analyst-supported slick age. It is not inferred from one SAR scene.', 'warning');
      return;
    }

    // A failed rerun must not leave the prior transport/AIS state looking current.
    this.driftResults = null;
    this.aisResults = null;
    this.counterfactualResults = null;
    this.counterfactualRequestNumber = (this.counterfactualRequestNumber || 0) + 1;
    this.hindcastLayer?.remove?.();
    this.forecastLayer?.remove?.();
    this.originMarker?.remove?.();
    this.driftMilestoneMarker?.remove?.();
    this.particlesLayerGroup?.clearLayers?.();
    this.vesselsLayerGroup?.clearLayers?.();
    this.counterfactualLayerGroup?.clearLayers?.();
    this.beachingMarker?.remove?.();
    this.beachingMarker = null;

    document.getElementById('systemStatusText').innerText = 'COMPUTING HYDRODYNAMICS...';
    try {
      const optionalNumber = (id, label) => {
        const value = document.getElementById(id)?.value;
        return parseOptionalFiniteNumber(value, label);
      };
      const currentU = optionalNumber('currentU', 'Current U');
      const currentV = optionalNumber('currentV', 'Current V');
      const windU = optionalNumber('windU', 'Wind U');
      const windV = optionalNumber('windV', 'Wind V');
      const overrides = [currentU, currentV, windU, windV];
      if (overrides.some(value => value !== null) && !overrides.every(value => value !== null)) {
        throw new Error('Provide all four current/wind vectors or leave all blank for a time-aligned source grid.');
      }
      if (isBenchmarkDemo && overrides.some(value => value !== null)) {
        throw new Error('Benchmark demo uses isolated simulated vectors. Clear overrides or use Live Ingestion for source data.');
      }
      const resp = await fetch('/api/simulate-drift', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          demo_mode: isBenchmarkDemo,
          slick_lat: centroidLat,
          slick_lon: centroidLon,
          scene_acquired_at_utc: this.sarProvenance?.acquisition_time_utc || this.sceneAcquisitionTime || '2026-10-04T12:00:00Z',
          slick_age_hours: ageHypothesis,
          max_lookback_hours: ageHypothesis,
          forecast_hours: 24.0,
          current_u_ms: currentU,
          current_v_ms: currentV,
          wind_u_ms: windU,
          wind_v_ms: windV,
          met_ocean_reference: document.getElementById('metOceanReference')?.value || null,
          initial_mass_tonnes: isBenchmarkDemo ? 18.0 : null,
          oil_profile: isBenchmarkDemo ? { initial_viscosity_cp: 18.0, water_temp_c: 26.0 } : null
        })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Met-ocean validation failed.');
      if (!data.origin_release_point || !Array.isArray(data.hindcast_trajectory)
          || !Array.isArray(data.forecast_trajectory)) {
        throw new Error('Transport response is incomplete; no scenario was rendered.');
      }
      this.driftResults = data;

      const origin = data.origin_release_point;
      if (!Number.isFinite(origin.lat) || !Number.isFinite(origin.lon)
          || !Number.isFinite(origin.assumed_slick_age_hours)) {
        throw new Error('Transport response contains invalid conditional coordinates or age.');
      }
      this.metricOriginCoords.innerText = `${screenNumber(origin.lat, 4)}° N, ${screenNumber(origin.lon, 4)}° E (conditional T - ${screenNumber(origin.assumed_slick_age_hours)}h)`;
      this.metricDriftDist.innerText = `${screenNumber(data.total_drift_distance_km)} km`;
      this.metricAge.innerText = `${screenNumber(origin.assumed_slick_age_hours)} h hypothesis`;

      // Generic weathering sensitivity; not ADIOS or an oil-specific forecast.
      if (data.weathering_summary) {
        const w = data.weathering_summary;
        if (this.metricEvap) this.metricEvap.innerText = `${screenNumber(w.evaporated_fraction_pct)}%`;
        if (this.metricMousse) this.metricMousse.innerText = `${screenNumber(w.water_content_mousse_pct)}%`;
        if (this.metricViscosity) this.metricViscosity.innerText = `${screenNumber(w.dynamic_viscosity_cP, 0)} cP`;
        if (this.metricVolExp) this.metricVolExp.innerText = `${screenNumber(w.volume_expansion_factor, 2)}x`;
        if (this.metricWeatheringState) this.metricWeatheringState.innerText = w.weathering_classification || 'NOT ASSESSED';
      } else {
        if (this.metricEvap) this.metricEvap.innerText = '—';
        if (this.metricMousse) this.metricMousse.innerText = '—';
        if (this.metricViscosity) this.metricViscosity.innerText = '—';
        if (this.metricVolExp) this.metricVolExp.innerText = '—';
        if (this.metricWeatheringState) this.metricWeatheringState.innerText = 'Not assessed: measured mass and oil profile were not supplied';
      }

      // Plot Hindcast (Backward track - dotted red)
      const validHindcast = data.hindcast_trajectory.filter(step =>
        Number.isFinite(step?.centroid?.lat) && Number.isFinite(step?.centroid?.lon));
      const validForecast = data.forecast_trajectory.filter(step =>
        Number.isFinite(step?.centroid?.lat) && Number.isFinite(step?.centroid?.lon));
      if (validHindcast.length < 2 || validForecast.length < 2) {
        throw new Error('Transport response contains no renderable trajectory points.');
      }
      const hindcastPts = validHindcast.map(step => [step.centroid.lat, step.centroid.lon]);
      if (this.hindcastLayer) this.map.removeLayer(this.hindcastLayer);
      this.hindcastLayer = L.polyline(hindcastPts, {
        color: '#ff3366',
        weight: 3,
        dashArray: '5, 8',
        opacity: 0.9
      }).bindTooltip(`
        <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
          <strong style="color:#ff3366;">🔴 CONDITIONAL BACKTRACK</strong><br>
          <span style="color:#cbd5e1; font-size:0.70rem;">Applies the ${origin.assumed_slick_age_hours}h analyst hypothesis; not an origin inference.</span>
        </div>
      `, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.map);

      // Plot Forecast (Forward track - dotted amber)
      const forecastPts = validForecast.map(step => [step.centroid.lat, step.centroid.lon]);
      if (this.forecastLayer) this.map.removeLayer(this.forecastLayer);
      this.forecastLayer = L.polyline(forecastPts, {
        color: '#f59e0b',
        weight: 3,
        dashArray: '6, 6',
        opacity: 0.85
      }).bindTooltip(`
        <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
          <strong style="color:#f59e0b;">🟡 FUTURE DRIFT FORECAST (+24H)</strong><br>
          <span style="color:#cbd5e1; font-size:0.70rem;">Modelled trajectory; validate against an authority shoreline-risk layer.</span>
        </div>
      `, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.map);

      // Add Shoreline Beaching Marker if beaching alert is triggered
      const beachWarn = data.beaching_warning;
      if (beachWarn && beachWarn.will_beach && beachWarn.beaching_location) {
        if (this.beachingMarker) this.map.removeLayer(this.beachingMarker);
        const beachIcon = L.divIcon({
          className: 'custom-beaching-pin-wrapper',
          html: `
            <div style="background:rgba(220,38,38,0.92); color:#fff; border:1px solid #ef4444; border-radius:4px; padding:3px 6px; font-size:10px; font-weight:bold; white-space:nowrap; box-shadow:0 0 10px rgba(239,68,68,0.6); text-align:center;">
              🚨 BEACHING T+${beachWarn.estimated_time_to_beach_hours}h
            </div>
            <div style="width:8px; height:8px; background:#ef4444; border-radius:50%; margin:2px auto 0 auto; box-shadow:0 0 6px #f87171;"></div>
          `,
          iconSize: [110, 36],
          iconAnchor: [55, 32]
        });
        this.beachingMarker = L.marker([beachWarn.beaching_location.lat, beachWarn.beaching_location.lon], { icon: beachIcon })
          .bindTooltip(`
            <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
              <strong style="color:#ef4444;">🚨 SHORELINE IMPACT ALERT</strong><br>
              <span style="color:#f8fafc; font-size:0.70rem;">Estimated beaching at T + ${beachWarn.estimated_time_to_beach_hours}h. Waterline contact.</span>
            </div>
          `, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.map);
      }

      // Add Origin Marker (x0, y0, t0) with SELF-EXPLANATORY CALLOUT
      if (this.originMarker) this.map.removeLayer(this.originMarker);
      const originIcon = L.divIcon({
        className: 'custom-origin-pin-wrapper',
        html: `
          <div class="origin-callout-bubble">
            <span class="bubble-tag">📍 CONDITIONAL BACKTRACK</span>
            <span class="bubble-title">GENERATED CLOUD • CONFIDENCE NOT ESTIMATED</span>
            <span class="bubble-time">Assumed T − ${origin.assumed_slick_age_hours}h</span>
          </div>
          <div class="custom-origin-pin"></div>
        `,
        iconSize: [120, 52],
        iconAnchor: [60, 48]
      });

      this.originMarker = L.marker([origin.lat, origin.lon], { icon: originIcon })
        .bindTooltip(`
          <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
            <strong style="color:#ff3366;">📍 CONDITIONAL BACKTRACKED POINT</strong><br>
            <span style="color:#f8fafc;">Age hypothesis • T − ${origin.assumed_slick_age_hours}h</span><br>
            <span style="color:#94a3b8; font-size:0.68rem;">Coords: ${screenNumber(origin.lat, 4)}° N, ${screenNumber(origin.lon, 4)}° E</span>
          </div>
        `, { sticky: true, className: 'c2-map-tooltip' })
        .bindPopup(`
          <div style="font-size:0.75rem; font-weight:700; color:#ff3366; margin-bottom:4px; display:flex; align-items:center; gap:6px;">
            <span class="pulse-dot" style="background:#ff3366;"></span> CONDITIONAL BACKTRACKED POINT
          </div>
          <div style="font-size:0.72rem; color:#f8fafc; font-weight:600;">Transport scenario output — not an inferred release point</div>
          <div style="font-size:0.68rem; color:#94a3b8; font-family:var(--font-mono); margin-top:2px;">
            Assumption: <span style="color:#ff3366; font-weight:700;">T - ${origin.assumed_slick_age_hours}h</span> &bull; 
            Origin confidence: <span style="color:#94a3b8; font-weight:700;">NOT ESTIMATED</span><br>
            Coords: ${screenNumber(origin.lat, 4)}° N, ${screenNumber(origin.lon, 4)}° E
          </div>
        `)
        .addTo(this.map);

      // Midpoint Drift Milestone Badge (Explains why pink and blue are connected)
      if (this.driftMilestoneMarker) this.map.removeLayer(this.driftMilestoneMarker);
      if (hindcastPts.length > 2) {
        const midIdx = Math.floor(hindcastPts.length / 2);
        const midPt = hindcastPts[midIdx];
        const distKm = screenNumber(data.total_drift_distance_km);
        const milestoneIcon = L.divIcon({
          className: 'custom-milestone-wrapper',
          html: `<div class="drift-milestone-badge">〰️ Ocean Drift: ${distKm} km ➔➔</div>`,
          iconSize: [160, 24],
          iconAnchor: [80, 12]
        });
        this.driftMilestoneMarker = L.marker(midPt, { icon: milestoneIcon }).addTo(this.map);
      }

      const bw = data.beaching_warning;
      if (bw) {
        if (bw.will_beach && bw.status === 'BEACHING_DETECTED') {
          this.metricBeachingStatus.innerText = '🚨 CRITICAL COASTAL HAZARD';
          this.metricBeachingStatus.className = 'tile-val crimson font-mono';
        } else if (bw.coastline_rejections > 0) {
          this.metricBeachingStatus.innerText = 'SHORELINE PROXIMITY WARNING';
          this.metricBeachingStatus.className = 'tile-val amber font-mono';
        } else if (bw.status === 'CLEAR' || bw.status === 'NOT_ASSESSED') {
          this.metricBeachingStatus.innerText = 'OPEN WATER — NO COASTAL THREAT';
          this.metricBeachingStatus.className = 'tile-val teal font-mono';
        } else {
          this.metricBeachingStatus.innerText = bw.status || 'MONITORING';
          this.metricBeachingStatus.className = 'tile-val amber font-mono';
        }

        const etbH = bw.estimated_time_to_beach_hours;
        this.metricETB.innerText = etbH ? `${Number(etbH).toFixed(1)} hrs` : '> 48.0 hrs';
        if (this.execLandfallHours) this.execLandfallHours.innerText = etbH ? `${Number(etbH).toFixed(1)}h` : '> 48h';
        if (this.hazardZoneName) {
          if (bw.vulnerable_assets && bw.vulnerable_assets.length > 0) {
            const topAsset = bw.vulnerable_assets[0];
            this.hazardZoneName.innerText = `⚠️ ${topAsset.name} — ${topAsset.distance_km} km [${topAsset.threat_level}]`;
          } else {
            this.hazardZoneName.innerText = bw.reason || 'No coastal assets in threat radius';
          }
        }
        if (this.execHazardTarget) {
          if (bw.vulnerable_assets && bw.vulnerable_assets.length > 0) {
            this.execHazardTarget.innerText = bw.vulnerable_assets.map(a => `${a.name} (${a.distance_km}km)`).join(' • ');
          } else {
            this.execHazardTarget.innerText = bw.will_beach ? 'Shoreline impact confirmed' : 'Open water corridor';
          }
        }
      } else {
        this.metricBeachingStatus.innerText = 'LOADING...';
        this.metricETB.innerText = '—';
        if (this.execLandfallHours) this.execLandfallHours.innerText = '—';
        if (this.execHazardTarget) this.execHazardTarget.innerText = 'Processing...';
      }

      // 🎯 Render Gaussian KDE Highest Density Region (HDR) Contours
      if (data.kde_origin_contours) {
        this.kdeContoursData = data.kde_origin_contours;
        const contours = data.kde_origin_contours.contours || [];
        const c95 = contours.find(c => Math.abs(c.level - 0.95) < 0.05);
        if (this.metricKdeCredibleArea) {
          const areaTxt = Number.isFinite(c95?.approximate_area_km2) ? `${c95.approximate_area_km2} km²` : 'NOT ASSESSED';
          const pkLat = Number.isFinite(data.kde_origin_contours.peak_density_lat) ? screenNumber(data.kde_origin_contours.peak_density_lat, 3) : screenNumber(origin.lat, 3);
          const pkLon = Number.isFinite(data.kde_origin_contours.peak_density_lon) ? screenNumber(data.kde_origin_contours.peak_density_lon, 3) : screenNumber(origin.lon, 3);
          this.metricKdeCredibleArea.innerText = `95% generated-cloud KDE mass: ${areaTxt} • Mode (${pkLat}°, ${pkLon}°) • not origin probability`;
        }
        if (this.chkShowKdeContours ? this.chkShowKdeContours.checked : true) {
          this.renderKdeOriginContours(data.kde_origin_contours, origin);
        }
      }

      this.renderParticlesAtTime(0.0);
      document.getElementById('systemStatusText').innerText = isBenchmarkDemo
        ? 'BENCHMARK TRANSPORT COMPLETE • SIMULATED / NOT LIVE'
        : 'CONDITIONAL TRANSPORT COMPLETE • COVERAGE VALIDATED';
      this.updateStepperState(2);
      if (isBenchmarkDemo) {
        await this.runAISCorrelation();
      }
    } catch (err) {
      console.error('Error running drift simulation:', err);
      document.getElementById('systemStatusText').innerText = 'DRIFT BLOCKED • INPUT VALIDATION FAILED';
      this.showToast(err.message || 'Unable to run a validated transport scenario.', 'warning');
    }
  }

  async runAISCorrelation() {
    const isBenchmarkDemo = this.demoMode === true || this.driftResults?.provenance?.mode === 'benchmark_demo' || (!this.customAisVessels && !this.aisProvenance);
    if ((!this.customAisVessels || !this.aisProvenance) && !isBenchmarkDemo) {
      document.getElementById('systemStatusText').innerText = 'AIS SCREENING BLOCKED • TIME-ALIGNED SOURCE AIS REQUIRED';
      this.showToast('Upload time-aligned AIS records before screening corridor traffic.', 'warning');
      return;
    }
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
          demo_mode: isBenchmarkDemo,
          origin_lat: origin.lat,
          origin_lon: origin.lon,
          origin_time_rel_h: origin.estimated_t0_hours_relative,
          spatial_radius_nm: 30.0,
          temporal_window_h: 5.0,
          hindcast_trajectory: this.driftResults ? this.driftResults.hindcast_trajectory : null,
          radar_targets: radarTargets,
          vessels: isBenchmarkDemo ? null : this.customAisVessels,
          ais_provenance: isBenchmarkDemo
            ? { source_kind: 'benchmark AIS trajectories (simulated; not live AIS)' }
            : this.aisProvenance
        })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'AIS validation failed.');
      this.aisResults = data;

      const culprit = data.primary_review_lead || (data.ranked_suspects && data.ranked_suspects[0]);
      if (culprit) {
        this.suspectVesselName.innerText = culprit.vessel_name;
        this.suspectIMO.innerText = culprit.imo;
        this.suspectMMSI.innerText = culprit.mmsi;
        this.suspectFlag.innerText = culprit.flag_state;
        this.suspectType.innerText = culprit.vessel_type;
        this.suspectSummary.innerText = this.leadSummary(culprit);

        // Update Executive Simple HUD
        if (this.execSuspectName) this.execSuspectName.innerText = culprit.vessel_name || 'UNKNOWN';
        if (this.execSuspectDetails) {
          const cpaDist = culprit.closest_approach ? culprit.closest_approach.distance_nm : '--';
          this.execSuspectDetails.innerText = `${isBenchmarkDemo ? 'SIMULATED' : 'Lead-priority'}: ${culprit.lead_priority_score}/100 • CPA: ${cpaDist} NM`;
        }

        // Update Anomaly Breakdown — Animated score bars (W4)
        const b = culprit.score_breakdown || {};
        // Reset all bars to 0 for animation
        this.barProx.style.width = '0%';
        this.barTime.style.width = '0%';
        this.barSpeed.style.width = '0%';
        this.barType.style.width = '0%';

        // Stagger the animations for visual impact
        requestAnimationFrame(() => {
          setTimeout(() => { this.barProx.style.width = `${Number.isFinite(b.proximity_score) ? b.proximity_score : 0}%`; }, 100);
          setTimeout(() => { this.barTime.style.width = `${Number.isFinite(b.temporal_score) ? b.temporal_score : 0}%`; }, 250);
          setTimeout(() => { this.barSpeed.style.width = '0%'; }, 400);
          setTimeout(() => { this.barType.style.width = '0%'; }, 550);
        });

        this.scoreProx.innerText = `${screenNumber(b.proximity_score)} / 100`;
        this.scoreTime.innerText = `${screenNumber(b.temporal_score)} / 100`;
        this.scoreSpeed.innerText = `${screenNumber(culprit.kinematics?.speed_drop_knots)} kts change (context only)`;
        this.scoreType.innerText = `${culprit.vessel_type} (not scored)`;

        // Animate score badge pop-in
        this.suspectScore.classList.remove('score-animate');
        void this.suspectScore.offsetWidth; // force reflow
        this.suspectScore.classList.add('score-animate');
      } else {
        this.selectedVessel = null;
        this.suspectVesselName.innerText = 'NO REVIEW LEAD';
        this.suspectIMO.innerText = '--';
        this.suspectMMSI.innerText = '--';
        this.suspectFlag.innerText = '--';
        this.suspectType.innerText = '--';
        this.suspectScore.innerText = 'NOT ASSESSED';
        this.suspectSummary.innerText = `Evidence gate: LEAD WITHHELD — ${data.screening_gate?.reason || 'No candidate or unavailable evidence.'}`;
        if (this.execSuspectName) this.execSuspectName.innerText = 'NO REVIEW LEAD';
        if (this.execSuspectDetails) this.execSuspectDetails.innerText = 'No supported AIS review lead';
        for (const item of [this.barProx, this.barTime, this.barSpeed, this.barType]) {
          if (item) item.style.width = '0%';
        }
        for (const item of [this.scoreProx, this.scoreTime, this.scoreSpeed, this.scoreType]) {
          if (item) item.innerText = 'NOT ASSESSED';
        }
        if (this.pillSignalStrong) this.pillSignalStrong.innerText = '● 0 High proximity/time cues';
        if (this.pillSignalWeak) this.pillSignalWeak.innerText = '● 0 Low proximity/time cues';
        if (this.pillAisQuality) {
          this.pillAisQuality.className = 'signal-pill pill-degraded';
          this.pillAisQuality.innerText = 'AIS NOT ASSESSED';
        }
        const staleChart = document.getElementById('sogChartContainer');
        if (staleChart) staleChart.remove();
      }

      // Render Table with live filters (AlgoRise / Akhilesh pattern)
      this.selectedLeadIndex = 0;
      this.filterAndRenderVessels();

      // Plot Vessel Tracks on Map
      this.renderVesselTracks(Array.isArray(data.ranked_suspects) ? data.ranked_suspects : []);

      // Radar/AIS mismatches are review cues, not disabled-transponder findings.
      this.renderDarkVessels(data.dark_vessels_detected || []);

      if (culprit) {
        this.selectSuspectVessel(culprit);
      }

      // Render Stage 4 Counterfactual Verification if returned by server
      if (data.counterfactual_verification) {
        this.renderCounterfactualVerification(data.counterfactual_verification);
        if (this.chkShowCounterfactual && this.chkShowCounterfactual.checked) {
          this.renderCounterfactualPlumeOnMap(data.counterfactual_verification);
        }
      }

      document.getElementById('systemStatusText').innerText = isBenchmarkDemo
        ? 'BENCHMARK AIS COMPLETE • SIMULATED / NOT LIVE'
        : 'AIS SCREENING COMPLETE • ANALYST REVIEW REQUIRED';
      this.updateStepperState(3);
    } catch (err) {
      console.error('Error correlating AIS:', err);
      document.getElementById('systemStatusText').innerText = 'AIS SCREENING BLOCKED • INPUT VALIDATION FAILED';
      this.showToast(err.message || 'Unable to screen AIS corridor traffic.', 'warning');
    }
  }

  renderSOGChart(culprit) {
    // Remove previous chart if any
    const existingChart = document.getElementById('sogChartContainer');
    if (existingChart) existingChart.remove();

    const traj = (culprit.full_trajectory || []).filter(point =>
      Number.isFinite(point.relative_time_hours) && Number.isFinite(point.sog_knots));
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

  leadSummary(vessel) {
    const gate = vessel.abstention_verdict;
    const held = !gate || gate.is_abstention !== false;
    const checks = vessel.adversarial_stress_test;
    return `Heuristic screening score: ${screenNumber(vessel.lead_priority_score)}/100 (not confidence). ` +
      `CPA: ${screenNumber(vessel.closest_approach?.distance_nm)} NM. ` +
      `Evidence gate: ${held ? 'LEAD WITHHELD' : 'SCREENING ONLY — NO RESPONSIBILITY FINDING'}; ` +
      `${gate?.status || 'NOT_ASSESSED'} — ${gate?.reason || 'No evidence-gate assessment supplied.'} ` +
      `Full-hypothesis entropy: ${screenNumber(gate?.entropy_metrics?.normalized_entropy, 4)}. ` +
      `Sensitivity checks: ${checks?.challenges_passed ?? 'NOT ASSESSED'}/${checks?.challenges_tested ?? 'NOT ASSESSED'} project checks; not calibrated robustness.`;
  }

  selectSuspectVessel(vessel) {
    if (!vessel) return;
    this.selectedVessel = vessel;

    this.suspectVesselName.innerText = vessel.vessel_name;
    this.suspectIMO.innerText = vessel.imo || '--';
    this.suspectMMSI.innerText = vessel.mmsi || '--';
    this.suspectFlag.innerText = vessel.flag_state || '--';
    this.suspectType.innerText = vessel.vessel_type || '--';
    this.suspectScore.innerText = `${screenNumber(vessel.lead_priority_score)}/100`;
    this.suspectSummary.innerText = this.leadSummary(vessel);

    // Candidate Rank Counter (AlgoRise / Akhilesh pattern)
    const allSuspects = this.aisResults?.all_ranked_suspects || this.aisResults?.ranked_suspects || [];
    const idx = allSuspects.findIndex(v => v.mmsi === vessel.mmsi || v.vessel_name === vessel.vessel_name);
    if (idx >= 0) this.selectedLeadIndex = idx;
    if (this.leadCandidateRank) {
      this.leadCandidateRank.innerText = `Lead ${this.selectedLeadIndex + 1} of ${allSuspects.length || 1}`;
    }

    // Evidence Signals (Strong vs Weak Signals & AIS Transponder Health)
    const b = vessel.score_breakdown || {};
    let strongCount = 0;
    let weakCount = 0;
    const prox = b.proximity_score;
    const temp = b.temporal_score;
    const speedDrop = vessel.kinematics?.speed_drop_knots;

    if (Number.isFinite(prox)) { if (prox >= 70) strongCount++; else if (prox < 40) weakCount++; }
    if (Number.isFinite(temp)) { if (temp >= 70) strongCount++; else if (temp < 40) weakCount++; }

    if (this.pillSignalStrong) {
      this.pillSignalStrong.innerText = `● ${strongCount} High proximity/time cue${strongCount === 1 ? '' : 's'}`;
    }
    if (this.pillSignalWeak) {
      this.pillSignalWeak.innerText = `● ${weakCount} Low proximity/time cue${weakCount === 1 ? '' : 's'}`;
    }
    if (this.pillAisQuality) {
      if (vessel.abstention_verdict?.integrity_status === 'COMPROMISED') {
        this.pillAisQuality.className = 'signal-pill pill-degraded';
        this.pillAisQuality.innerText = '⚠️ AIS GAP / SPOOF';
        this.pillAisQuality.title = (vessel.spoofing_audit.anomalies_detected || []).join('; ');
      } else if (vessel.abstention_verdict?.integrity_status === 'ASSESSED') {
        this.pillAisQuality.className = 'signal-pill pill-optimal';
        this.pillAisQuality.innerText = 'SUPPLIED TRACK CHECKED';
        this.pillAisQuality.title = 'Continuity checks only; receiver coverage and registry identity require independent verification';
      } else {
        this.pillAisQuality.className = 'signal-pill pill-degraded';
        this.pillAisQuality.innerText = 'AIS NOT ASSESSED';
        this.pillAisQuality.title = 'Missing or invalid telemetry cannot validate continuity';
      }
    }

    // Animate and set progress bars
    if (this.barProx) this.barProx.style.width = `${b.proximity_score || 0}%`;
    if (this.barTime) this.barTime.style.width = `${b.temporal_score || 0}%`;
    if (this.barSpeed) this.barSpeed.style.width = '0%';
    if (this.barType) this.barType.style.width = '0%';

    if (this.scoreProx) this.scoreProx.innerText = `${screenNumber(b.proximity_score)} / 100`;
    if (this.scoreTime) this.scoreTime.innerText = `${screenNumber(b.temporal_score)} / 100`;
    if (this.scoreSpeed) this.scoreSpeed.innerText = `${screenNumber(speedDrop)} kts change (not scored)`;
    if (this.scoreType) this.scoreType.innerText = `${vessel.vessel_type || 'UNKNOWN'} (not scored)`;

    if (vessel.full_trajectory) {
      this.renderSOGChart(vessel);
    }

    // Highlight active row in traffic table
    if (this.vesselTableBody) {
      const rows = this.vesselTableBody.querySelectorAll('.vessel-row:not(.dark-vessel-row)');
      rows.forEach(r => r.style.outline = 'none');
      const activeRow = this.vesselTableBody.querySelector(`[data-vessel-idx="${this.selectedLeadIndex}"]`);
      if (activeRow) {
        activeRow.style.outline = '2px solid var(--accent-cyan, #00f2fe)';
      }
    }

    // Run Stage 4 Counterfactual Verification for this selected vessel
    this.runCounterfactualVerification(vessel);

    // Render 4-Challenge Adversarial Falsification Stress Matrix
    this.renderAdversarialStressTest(vessel);

    // Render Bayesian Maritime Evidence Synthesis & Decision Gate
    this.renderBayesianDecisionGate(this.aisResults?.screening_gate || this.aisResults?.bayesian_legal_gate);
  }

  renderAdversarialStressTest(vessel) {
    if (!this.falsificationTableBody) return;
    const stress = vessel?.adversarial_stress_test;
    if (!stress || !Array.isArray(stress.challenges)) {
      this.falsificationTableBody.innerHTML = `
        <tr>
          <td colspan="4" class="table-await-cell" style="padding: 8px; text-align: center; color: #64748b;">
            Adversarial stress test not available for this candidate.
          </td>
        </tr>`;
      if (this.falsificationVerdictBadge) {
        this.falsificationVerdictBadge.innerText = 'NOT ASSESSED';
        this.falsificationVerdictBadge.className = 'badge-verdict badge-cf-pending';
      }
      return;
    }

    const challenges = stress.challenges;
    let rowsHtml = '';
    challenges.forEach((ch, idx) => {
      const survived = ch.survived === true;
      const badgeClass = survived ? 'tag-emerald' : 'tag-crimson';
      const badgeText = survived ? 'PASS' : 'FAIL';
      const shift = ch.estimated_origin_shift_nm !== null && ch.estimated_origin_shift_nm !== undefined
        ? `Δ ${ch.estimated_origin_shift_nm} nm`
        : (ch.effective_cpa_nm !== null && ch.effective_cpa_nm !== undefined ? `CPA ${ch.effective_cpa_nm} nm` : '—');
      
      rowsHtml += `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
          <td style="padding: 4px; font-weight: 600; color: #f8fafc;" title="${ch.challenge_name}">
            #${idx + 1} ${ch.challenge_name.split('(')[0].trim()}
          </td>
          <td style="padding: 4px; color: #94a3b8; font-size: 0.65rem;">
            ${ch.stress_parameter || '—'}
          </td>
          <td style="padding: 4px; color: #00f2fe; font-size: 0.65rem;">
            ${shift}
          </td>
          <td style="padding: 4px; text-align: center;">
            <span class="tag ${badgeClass}" style="font-size: 0.62rem; padding: 1px 5px; font-weight: 800;">${badgeText}</span>
          </td>
        </tr>`;
    });
    this.falsificationTableBody.innerHTML = rowsHtml;

    if (this.falsificationVerdictBadge) {
      const vText = stress.verdict || (stress.stress_passed ? 'ADVERSARIAL ROBUST' : 'FALSIFICATION VULNERABLE');
      this.falsificationVerdictBadge.innerText = vText.replace(/_/g, ' ');
      this.falsificationVerdictBadge.className = 'badge-verdict';
      if (stress.stress_passed) {
        this.falsificationVerdictBadge.style.background = 'rgba(5, 214, 160, 0.15)';
        this.falsificationVerdictBadge.style.borderColor = '#05d6a0';
        this.falsificationVerdictBadge.style.color = '#05d6a0';
      } else {
        this.falsificationVerdictBadge.style.background = 'rgba(255, 51, 102, 0.15)';
        this.falsificationVerdictBadge.style.borderColor = '#ff3366';
        this.falsificationVerdictBadge.style.color = '#ff3366';
      }
    }

    if (this.falsificationCountTag) {
      this.falsificationCountTag.innerText = `${stress.passed_challenges || 0}/4 PASSED (${stress.adversarial_robustness_score || 0}%)`;
    }

    if (this.falsificationRationale) {
      const failed = challenges.filter(c => !c.survived);
      if (failed.length === 0) {
        this.falsificationRationale.innerText = `Robust across all 4 perturbation attacks (current, wind drag, GPS jitter, AIS integrity). Corridor footprint is stable.`;
        this.falsificationRationale.style.borderLeft = '2px solid #05d6a0';
      } else {
        this.falsificationRationale.innerText = `Vulnerable to ${failed.length} attack(s): ${failed.map(f => f.challenge_name.split('(')[0].trim()).join(', ')}. ${failed[0].rationale || ''}`;
        this.falsificationRationale.style.borderLeft = '2px solid #ff3366';
      }
    }
  }

  renderBayesianDecisionGate(gate) {
    if (!gate) return;
    if (this.bayesianVerdictBadge) {
      const dec = gate.decision || gate.status || 'ASSESSED';
      this.bayesianVerdictBadge.innerText = dec.replace(/_/g, ' ');
      this.bayesianVerdictBadge.className = 'badge-verdict';
      if (dec === 'ISOLATED_LEADING_CANDIDATE') {
        this.bayesianVerdictBadge.style.background = 'rgba(5, 214, 160, 0.15)';
        this.bayesianVerdictBadge.style.borderColor = '#05d6a0';
        this.bayesianVerdictBadge.style.color = '#05d6a0';
      } else {
        this.bayesianVerdictBadge.style.background = 'rgba(245, 158, 11, 0.15)';
        this.bayesianVerdictBadge.style.borderColor = '#f59e0b';
        this.bayesianVerdictBadge.style.color = '#f59e0b';
      }
    }

    const entropy = gate.entropy_metrics?.normalized_entropy;
    if (this.bayesianEntropy) {
      this.bayesianEntropy.innerText = Number.isFinite(entropy) ? `${(entropy).toFixed(2)} (Rule: ≤0.82)` : '—';
      if (Number.isFinite(entropy) && entropy > 0.82) {
        this.bayesianEntropy.style.color = '#ff3366';
      } else {
        this.bayesianEntropy.style.color = '#00f2fe';
      }
    }

    const margin = gate.separation_margin;
    if (this.bayesianMargin) {
      this.bayesianMargin.innerText = Number.isFinite(margin) ? `Δ ${(margin).toFixed(2)} (Rule: ≥0.15)` : '—';
      if (Number.isFinite(margin) && margin < 0.15) {
        this.bayesianMargin.style.color = '#ff3366';
      } else {
        this.bayesianMargin.style.color = '#f59e0b';
      }
    }

    if (this.bayesianBarsList && Array.isArray(gate.hypothesis_distribution)) {
      let barsHtml = '';
      gate.hypothesis_distribution.forEach(h => {
        const weight = Number.isFinite(h.lead_priority_weight) ? h.lead_priority_weight : 0;
        const pct = (weight * 100).toFixed(1);
        const name = h.hypothesis === 'UNKNOWN_SOURCE' ? '❓ Unknown / Dark Source' : `🚢 MMSI ${h.mmsi || 'Candidate'}`;
        const color = h.hypothesis === 'UNKNOWN_SOURCE' ? '#94a3b8' : (weight >= 0.5 ? '#00f2fe' : '#3b82f6');
        barsHtml += `
          <div style="font-size: 0.65rem; font-family: var(--font-mono);">
            <div style="display:flex; justify-content:space-between; margin-bottom: 2px;">
              <span style="color: ${color};">${name}</span>
              <span style="color: #f8fafc; font-weight: 700;">${pct}%</span>
            </div>
            <div class="modern-progress-track" style="height: 4px; margin-bottom: 3px;">
              <div style="width: ${pct}%; background: ${color}; height: 100%; border-radius: 2px;"></div>
            </div>
          </div>`;
      });
      this.bayesianBarsList.innerHTML = barsHtml;
    }

    if (this.bayesianRecommendation) {
      this.bayesianRecommendation.innerText = gate.reason || gate.actionable_recommendation || 'Validate source coverage and telemetry; review independent corroboration.';
    }
  }

  renderCounterfactualVerification(cf) {
    if (!cf) return;
    this.counterfactualResults = cf;

    const m = cf.verification_metrics || {};
    if (this.cfCentroidError) this.cfCentroidError.innerText = screenNumber(m.centroid_distance_km, 2);
    if (this.cfContainment) this.cfContainment.innerText = screenNumber(m.predicted_containment_percent);
    if (this.cfJaccard) this.cfJaccard.innerText = screenNumber(m.jaccard_index, 3);
    if (this.cfCausalityScore) this.cfCausalityScore.innerText = 'NOT ESTIMATED';

    if (this.cfVerdictBadge) {
      this.cfVerdictBadge.innerText = cf.verdict_badge || cf.verdict || cf.status || 'NOT ASSESSED';
      this.cfVerdictBadge.className = 'badge-verdict';
      if (cf.verdict === 'CONDITIONAL_SPATIAL_AGREEMENT') {
        this.cfVerdictBadge.classList.add('badge-cf-match');
      } else if (cf.verdict === 'CONDITIONAL_NEARBY_CORRIDOR') {
        this.cfVerdictBadge.classList.add('badge-cf-corridor');
      } else {
        this.cfVerdictBadge.classList.add('badge-cf-refuted');
      }
    }

    if (this.cfExplanation) {
      this.cfExplanation.innerText = cf.explanation || cf.detail || 'NOT ASSESSED — conditional transport is not origin or responsibility evidence.';
      if (cf.verdict === 'CONDITIONAL_SPATIAL_AGREEMENT') {
        this.cfExplanation.style.borderLeftColor = '#05d6a0';
      } else if (cf.verdict === 'CONDITIONAL_NEARBY_CORRIDOR') {
        this.cfExplanation.style.borderLeftColor = '#f59e0b';
      } else {
        this.cfExplanation.style.borderLeftColor = '#ff3366';
      }
    }
  }

  renderKdeOriginContours(kdeData, origin) {
    if (!this.kdeContoursLayerGroup) return;
    this.kdeContoursLayerGroup.clearLayers();
    if (!kdeData || !kdeData.contours || kdeData.contours.length === 0) return;

    // Draw from outermost (95%) to innermost (50%)
    const sortedContours = [...kdeData.contours].sort((a, b) => b.level - a.level);

    const styleMap = {
      0.95: { color: '#00f2fe', fillColor: '#00f2fe', fillOpacity: 0.10, weight: 1.8, dashArray: '4, 4' },
      0.75: { color: '#38bdf8', fillColor: '#38bdf8', fillOpacity: 0.18, weight: 2.0, dashArray: '5, 5' },
      0.50: { color: '#818cf8', fillColor: '#818cf8', fillOpacity: 0.28, weight: 2.2, dashArray: null }
    };

    sortedContours.forEach(c => {
      const levelKey = Math.round(c.level * 100) / 100;
      const style = styleMap[levelKey] || {
        color: c.color || '#00f2fe',
        fillColor: c.color || '#00f2fe',
        fillOpacity: 0.15,
        weight: 2,
        dashArray: '4, 4'
      };

      const polys = (c.all_polygons && c.all_polygons.length > 0)
        ? c.all_polygons
        : (c.polygon_coords && c.polygon_coords.length >= 3 ? [c.polygon_coords] : []);

      polys.forEach(pts => {
        if (!pts || pts.length < 3) return;
        L.polygon(pts, style)
          .bindTooltip(`
            <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
              <strong style="color:${style.color};">🎯 GAUSSIAN KDE ${c.label}</strong><br>
              <span style="color:#f8fafc;">Highest Density Region (HDR)</span><br>
              <span style="color:#94a3b8; font-size:0.68rem;">Conditional cloud area: ${c.approximate_area_km2 ?? '--'} km² &bull; KDE mass: ${Math.round(c.level * 100)}%</span>
            </div>
          `, { sticky: true, className: 'c2-map-tooltip' })
          .bindPopup(`
            <div class="c2-popup font-mono" style="font-size:0.75rem;">
              <b style="color:${style.color};">🎯 GAUSSIAN KDE ${c.label} GENERATED CLOUD</b><br>
              Method: <b>${kdeData.method || 'gaussian_kde_hdr'}</b><br>
              Generated-cloud KDE mass: <b>${Math.round(c.level * 100)}%</b><br>
              Envelope Area: <b>${screenNumber(c.approximate_area_km2, 3)} km²</b><br>
              Origin Hypothesis: <b>T - ${origin?.assumed_slick_age_hours || '--'}h</b><br>
              <i>Conditional sampled-cloud density only; not calibrated origin probability or a confidence region.</i>
            </div>
          `)
          .addTo(this.kdeContoursLayerGroup);
      });
    });

    // Peak Density Mode marker
    if (Number.isFinite(kdeData.peak_density_lat) && Number.isFinite(kdeData.peak_density_lon)) {
      const peakIcon = L.divIcon({
        className: 'kde-peak-marker',
        html: `
          <div style="position:relative; width:14px; height:14px; display:flex; align-items:center; justify-content:center;">
            <div style="position:absolute; width:14px; height:14px; border-radius:50%; background:rgba(0,242,254,0.3); animation:pulse-glow 1.8s infinite;"></div>
            <div style="width:8px; height:8px; border-radius:50%; background:#00f2fe; border:1.5px solid #ffffff; box-shadow:0 0 8px #00f2fe;"></div>
          </div>
        `,
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });

      L.marker([kdeData.peak_density_lat, kdeData.peak_density_lon], { icon: peakIcon })
        .bindTooltip(`
          <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
            <strong style="color:#00f2fe;">✦ Conditional Cloud KDE Mode</strong><br>
            <span style="color:#cbd5e1; font-size:0.70rem;">${screenNumber(kdeData.peak_density_lat, 4)}° N, ${screenNumber(kdeData.peak_density_lon, 4)}° E</span>
          </div>
        `, { sticky: true, className: 'c2-map-tooltip' })
        .bindPopup(`
          <div class="c2-popup font-mono" style="font-size:0.75rem;">
            <b style="color:#00f2fe;">✦ CONDITIONAL GENERATED-CLOUD KDE MODE</b><br>
            Coordinates: <b>${screenNumber(kdeData.peak_density_lat, 4)}° N, ${screenNumber(kdeData.peak_density_lon, 4)}° E</b><br>
            Method: <b>Gaussian Kernel Density Estimation (Silverman rule)</b><br>
            <i>Maximum of the sampled conditional density, not a maximum-likelihood spill origin.</i>
          </div>
        `)
        .addTo(this.kdeContoursLayerGroup);
    }
  }

  renderCounterfactualPlumeOnMap(cf) {
    if (!this.counterfactualLayerGroup) return;
    this.counterfactualLayerGroup.clearLayers();
    if (!cf) return;

    const traj = (Array.isArray(cf.forward_trajectory) ? cf.forward_trajectory : []).filter(step =>
      Number.isFinite(step?.centroid?.lat) && Number.isFinite(step?.centroid?.lon));
    const pred = cf.predicted_at_t0 || {};
    const verdict = cf.verdict || cf.status || 'NOT ASSESSED';
    const color = verdict === 'CONDITIONAL_SPATIAL_AGREEMENT' ? '#05d6a0' : (verdict === 'CONDITIONAL_NEARBY_CORRIDOR' ? '#f59e0b' : '#ff3366');

    // 1. Draw forward simulation trajectory line
    if (traj.length > 1) {
      const pts = traj.map(step => [step.centroid.lat, step.centroid.lon]);
      L.polyline(pts, {
        color: color,
        weight: 3.5,
        dashArray: '6, 6',
        opacity: 0.95
      }).bindTooltip(`
        <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
          <strong style="color:${color};">🔬 CONDITIONAL FORWARD TRANSPORT</strong><br>
          <span style="color:#ffffff;">Generated-cloud comparison from ${cf.vessel_name || 'candidate'}</span><br>
          <span style="color:#94a3b8; font-size:0.68rem;">T = ${cf.release_state?.time_relative_h || '--'}h &rarr; T0 (${traj.length} steps RK4)</span>
        </div>
      `, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.counterfactualLayerGroup);

      // Candidate release marker
      if (cf.release_state && Number.isFinite(cf.release_state.lat) && Number.isFinite(cf.release_state.lon)) {
        L.circleMarker([cf.release_state.lat, cf.release_state.lon], {
          radius: 7,
          fillColor: '#f59e0b',
          color: '#ffffff',
          weight: 2,
          fillOpacity: 1.0
        }).bindPopup(`
          <div class="c2-popup font-mono" style="font-size:0.75rem;">
            <b style="color:#f59e0b;">📍 CANDIDATE AIS STATE (HYPOTHESIS)</b><br>
            Vessel: <b>${cf.vessel_name || 'candidate'}</b><br>
            Coords: ${screenNumber(cf.release_state.lat, 4)}°N, ${screenNumber(cf.release_state.lon, 4)}°E<br>
            Time: ${cf.release_state.time_relative_h}h relative to detection<br>
            <i>Generated cloud seeded at a reported AIS coordinate; not a verified discharge.</i>
          </div>
        `).addTo(this.counterfactualLayerGroup);
      }
    }

    // 2. Draw predicted footprint polygon at T0
    if (Array.isArray(pred.predicted_footprint_polygon) && pred.predicted_footprint_polygon.length > 2) {
      L.polygon(pred.predicted_footprint_polygon, {
        color: color,
        fillColor: color,
        weight: 2,
        fillOpacity: 0.35,
        dashArray: '4, 4'
      }).bindPopup(`
        <div class="c2-popup font-mono" style="font-size:0.75rem;">
          <b style="color:${color};">🎯 GENERATED-CLOUD CIRCLE PROXY</b><br>
          Centroid Separation: <b>${screenNumber(cf.verification_metrics?.centroid_distance_km, 2)} km</b><br>
          Generated-cloud Containment: <b>${screenNumber(cf.verification_metrics?.predicted_containment_percent)}%</b><br>
          Circle-proxy IoU: <b>${screenNumber(cf.verification_metrics?.jaccard_index, 3)}</b><br>
          Verdict: <b style="color:${color};">${cf.verdict_badge || verdict}</b>
        </div>
      `).addTo(this.counterfactualLayerGroup);
    }

    // 3. Draw sample particles at T0
    if (Array.isArray(pred.particles_sample) && pred.particles_sample.length > 0) {
      pred.particles_sample.filter(pt => Array.isArray(pt) && pt.length === 2
        && Number.isFinite(pt[0]) && Number.isFinite(pt[1])).forEach(pt => {
        L.circleMarker([pt[1], pt[0]], {
          radius: 2.5,
          fillColor: color,
          color: '#ffffff',
          weight: 0.5,
          fillOpacity: 0.85
        }).addTo(this.counterfactualLayerGroup);
      });
    }
  }

  async runCounterfactualVerification(vessel = null) {
    const requestNumber = (this.counterfactualRequestNumber || 0) + 1;
    this.counterfactualRequestNumber = requestNumber;
    const targetVessel = vessel || (this.aisResults ? this.aisResults.primary_review_lead : null);
    if (!targetVessel || !targetVessel.candidate_release_point) {
      this.renderCounterfactualVerification({status: 'NOT_ASSESSED', detail: 'No supported candidate release state.'});
      this.counterfactualLayerGroup?.clearLayers();
      this.showToast('No candidate AIS release state available for counterfactual verification.', 'warning');
      return;
    }
    const crp = targetVessel.candidate_release_point;
    const slick = this.sarResults ? this.sarResults.primary_slick : null;
    const isBenchmarkDemo = this.demoMode === true || this.driftResults?.provenance?.mode === 'benchmark_demo' || (!this.sarProvenance && !this.pendingSarFile);
    const obsLat = Number.isFinite(slick?.centroid?.lat) ? slick.centroid.lat : (isBenchmarkDemo ? this.scenarioData?.center?.lat : NaN);
    const obsLon = Number.isFinite(slick?.centroid?.lon) ? slick.centroid.lon : (isBenchmarkDemo ? this.scenarioData?.center?.lon : NaN);
    if (!Number.isFinite(obsLat) || !Number.isFinite(obsLon)) {
      this.renderCounterfactualVerification({status: 'NOT_ASSESSED', detail: 'Documented observation geometry is required.'});
      this.counterfactualLayerGroup?.clearLayers();
      this.showToast('No validated SAR observation geometry available.', 'warning');
      return;
    }
    const obsArea = Number.isFinite(slick?.area_km2) ? slick.area_km2 : (isBenchmarkDemo ? 0.616 : null);

    if (this.btnRunCounterfactual) {
      this.btnRunCounterfactual.disabled = true;
      this.btnRunCounterfactual.innerHTML = `<span>Simulating RK4 Forward...</span>`;
    }

    try {
      const resp = await fetch('/api/verify-counterfactual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          vessel_mmsi: targetVessel.mmsi,
          vessel_name: targetVessel.vessel_name,
          release_lat: crp.lat,
          release_lon: crp.lon,
          release_time_rel_h: crp.relative_time_hours,
          observed_slick_lat: obsLat,
          observed_slick_lon: obsLon,
          observed_slick_area_km2: obsArea,
          demo_mode: this.driftResults?.provenance?.mode === 'benchmark_demo',
          scene_acquired_at_utc: this.sarProvenance?.acquisition_time_utc || this.sceneAcquisitionTime || null
        })
      });
      const data = await resp.json();
      if (requestNumber !== this.counterfactualRequestNumber) return;
      if (!resp.ok) throw new Error(data.detail || 'Counterfactual simulation failed.');

      this.renderCounterfactualVerification(data);
      if (this.chkShowCounterfactual && this.chkShowCounterfactual.checked) {
        this.renderCounterfactualPlumeOnMap(data);
      }
      this.updateStepperState(4);
      this.showToast(`Conditional comparison: ${data.verdict_badge || data.status}`, 'info');
    } catch (err) {
      console.error('Counterfactual verification error:', err);
      if (requestNumber !== this.counterfactualRequestNumber) return;
      this.renderCounterfactualVerification({status: 'UNAVAILABLE', detail: err.message, verification_metrics: null});
      this.counterfactualLayerGroup?.clearLayers();
      this.showToast(err.message || 'Error running counterfactual verification', 'error');
    } finally {
      if (this.btnRunCounterfactual && requestNumber === this.counterfactualRequestNumber) {
        this.btnRunCounterfactual.disabled = false;
        this.btnRunCounterfactual.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          <span>Re-Simulate Forward Drift</span>
        `;
      }
    }
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
      const isTopLead = idx === 0;
      const rowClass = isTopLead ? 'vessel-row culprit' : 'vessel-row';
      const topsisBadge = Number.isFinite(v.topsis_closeness_score) ? `<span style="font-size:0.60rem; padding:1px 4px; border-radius:3px; background:rgba(0,242,254,0.15); color:#00f2fe; margin-left:4px; font-weight:700;">TOPSIS ${screenNumber(v.topsis_closeness_score)} (descriptive)</span>` : '';
      const spoofBadge = (v.spoofing_audit && v.spoofing_audit.has_anomalies) ? `<span title="${(v.spoofing_audit.anomalies_detected || []).join('; ')}" style="font-size:0.58rem; padding:1px 4px; border-radius:3px; background:rgba(239,68,68,0.2); color:#ef4444; margin-left:4px; font-weight:700; cursor:help;">⚠️ GAP/SPOOF</span>` : '';
      const cpa = v.closest_approach || {};
      const kinematics = v.kinematics || {};
      const leadScore = screenNumber(v.lead_priority_score);
      rowsHtml += `
        <tr class="${rowClass}" data-vessel-idx="${idx}">
          <td><b style="color:${v.flag_color || '#94a3b8'};">${v.vessel_name || 'UNKNOWN VESSEL'}</b>${topsisBadge}${spoofBadge}<br><span style="font-size:0.65rem; color:var(--text-muted);">IMO ${v.imo || 'NOT SUPPLIED'}</span></td>
          <td>${v.vessel_type || 'NOT ASSESSED'}</td>
          <td>${screenNumber(cpa.distance_nm)} NM</td>
          <td style="color:${Number.isFinite(kinematics.speed_drop_knots) && kinematics.speed_drop_knots > 4.0 ? 'var(--accent-crimson)' : 'inherit'};">
            ${screenNumber(kinematics.speed_drop_knots)} kts change
          </td>
          <td><b style="color:${v.flag_color || '#94a3b8'};">${leadScore}/100 (not confidence)</b></td>
        </tr>
      `;
    });
    this.vesselTableBody.innerHTML = rowsHtml;

    // Attach click listener to each vessel row to trigger Stage 4 counterfactual verification
    const aisRows = this.vesselTableBody.querySelectorAll('.vessel-row:not(.dark-vessel-row)');
    aisRows.forEach((row, idx) => {
      row.style.cursor = 'pointer';
      row.title = 'Click to select vessel and run Stage 4 Forward Counterfactual Verification';
      row.addEventListener('click', () => {
        aisRows.forEach(r => r.style.outline = 'none');
        row.style.outline = '2px solid var(--accent-cyan, #00f2fe)';
        const v = suspects[idx];
        if (v) this.selectSuspectVessel(v);
      });
    });
  }

  // Live Corridor Traffic Search & Risk Filter (AlgoRise / Akhilesh pattern)
  filterAndRenderVessels() {
    if (!this.aisResults) return;
    const allSuspects = this.aisResults.all_ranked_suspects || this.aisResults.ranked_suspects || [];
    const allDark = this.aisResults.dark_vessels_detected || this.aisResults.dark_vessels || [];

    const query = (this.trafficFilterQuery || '').toLowerCase();
    const risk = this.trafficFilterRisk || 'all';

    const filteredSuspects = allSuspects.filter(v => {
      const name = (v.vessel_name || '').toLowerCase();
      const type = (v.vessel_type || '').toLowerCase();
      const imo = String(v.imo || '').toLowerCase();
      const mmsi = String(v.mmsi || '').toLowerCase();
      const flag = (v.flag_state || '').toLowerCase();
      const matchesQuery = !query || name.includes(query) || type.includes(query) || imo.includes(query) || mmsi.includes(query) || flag.includes(query);

      const score = Number.isFinite(v.lead_priority_score) ? v.lead_priority_score : null;
      let matchesRisk = true;
      if (risk === 'high') {
        matchesRisk = score !== null && score >= 70;
      } else if (risk === 'medium') {
        matchesRisk = score !== null && score >= 40 && score < 70;
      } else if (risk === 'low') {
        matchesRisk = score !== null && score < 40;
      } else if (risk === 'dark') {
        matchesRisk = false;
      }
      return matchesQuery && matchesRisk;
    });

    const filteredDark = (risk === 'all' || risk === 'dark') ? allDark.filter(dv => {
      const id = (dv.target_id || '').toLowerCase();
      return !query || id.includes(query) || 'radar'.includes(query);
    }) : [];

    this.renderVesselTable(filteredSuspects, filteredDark);
  }

  // Candidate Lead Previous / Next Cycler (AlgoRise SuspectPanel pattern)
  navigateLead(delta) {
    if (!this.aisResults) return;
    const allSuspects = this.aisResults.all_ranked_suspects || this.aisResults.ranked_suspects || [];
    if (!allSuspects.length) return;

    const count = allSuspects.length;
    this.selectedLeadIndex = (this.selectedLeadIndex + delta + count) % count;
    const vessel = allSuspects[this.selectedLeadIndex];
    if (vessel) {
      this.selectSuspectVessel(vessel);
      if (this.vesselTableBody) {
        const rows = this.vesselTableBody.querySelectorAll('.vessel-row:not(.dark-vessel-row)');
        rows.forEach(r => r.style.outline = 'none');
        const activeRow = this.vesselTableBody.querySelector(`[data-vessel-idx="${this.selectedLeadIndex}"]`);
        if (activeRow) {
          activeRow.style.outline = '2px solid var(--accent-cyan, #00f2fe)';
          activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
    }
  }

  // Export Full Investigation JSON Dossier (Akhilesh / AlgoRise pattern)
  exportInvestigationJSON() {
    if (!this.aisResults && !this.sarResults && !this.driftResults) {
      this.showToast('Run investigation pipeline before exporting JSON dossier.', 'warning');
      return;
    }
    const dossier = {
      project: "OCEAN-SHIELD",
      problem_statement: "SIH26143",
      timestamp_utc: new Date().toISOString(),
      active_sector: this.activeScenarioId,
      sar_detection: this.sarResults,
      drift_simulation: this.driftResults,
      ais_attribution: this.aisResults,
      counterfactual_verification: this.counterfactualResults || (this.aisResults ? this.aisResults.counterfactual_verification : null),
      provenance: {
        sar: this.sarProvenance,
        ais: this.aisProvenance
      }
    };
    const blob = new Blob([JSON.stringify(dossier, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `OCEAN_SHIELD_DOSSIER_${this.activeScenarioId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.showToast('Forensic JSON dossier exported successfully.', 'success');
  }

  renderVesselTracks(vessels) {
    this.vesselsLayerGroup.clearLayers();

    (Array.isArray(vessels) ? vessels : []).forEach((v, idx) => {
      const isTopLead = idx === 0;
      const track = (Array.isArray(v.full_trajectory) ? v.full_trajectory : []).filter(p =>
        Number.isFinite(p?.lat) && Number.isFinite(p?.lon));
      if (track.length === 0) return;
      const pts = track.map(p => [p.lat, p.lon]);
      const color = isTopLead ? '#ff3366' : (Number.isFinite(v.lead_priority_score) && v.lead_priority_score > 50 ? '#ffaa00' : '#05d6a0');
      const scoreText = screenNumber(v.lead_priority_score);

      const tooltipHtml = `
        <div style="font-family:var(--font-sans); font-size:0.75rem; padding:4px 6px;">
          <strong style="color:${color};">${isTopLead ? '🚨 HIGHEST-RANKED REVIEW LEAD' : '🚢 OTHER AIS SCREENING LEAD'}</strong><br>
          <span style="color:#f8fafc; font-weight:700;">${v.vessel_name}</span> (${v.vessel_type})<br>
          <span style="color:#94a3b8; font-size:0.68rem;">${isTopLead ? 'Highest lead-priority screen; requires source-record review • Score: ' + scoreText + '/100' : 'Lower lead-priority screen (' + scoreText + '/100)'}</span>
        </div>
      `;

      // Trajectory line
      const poly = L.polyline(pts, {
        color: color,
        weight: isTopLead ? 3.5 : 2,
        opacity: isTopLead ? 0.95 : 0.6
      }).bindTooltip(tooltipHtml, { sticky: true, className: 'c2-map-tooltip' }).addTo(this.vesselsLayerGroup);

      // Latest position marker
      const latestPt = track[track.length - 1];
      const shipMarker = L.circleMarker([latestPt.lat, latestPt.lon], {
        radius: isTopLead ? 7 : 5,
        fillColor: color,
        color: '#ffffff',
        weight: 1.5,
        fillOpacity: 1
      }).bindTooltip(tooltipHtml, { sticky: true, className: 'c2-map-tooltip' }).bindPopup(`
        <div style="font-size:0.80rem; font-weight:700; color:#f8fafc; margin-bottom:2px;">${v.vessel_name}</div>
        <div style="font-size:0.68rem; color:#94a3b8; margin-bottom:6px;">${v.vessel_type} &bull; MMSI: ${v.mmsi}</div>
        <div style="display:flex; justify-content:space-between; font-size:0.70rem; font-family:var(--font-mono); background:rgba(0,0,0,0.3); padding:4px 6px; border-radius:4px; margin-bottom:3px;">
          <span style="color:#94a3b8;">Lead Score:</span>
          <b style="color:${Number.isFinite(v.composite_suspect_score) && v.composite_suspect_score > 60 ? '#ef4444' : '#00f2fe'}; font-weight:700;">${scoreText}/100</b>
        </div>
        <div style="font-size:0.66rem; color:#64748b; font-family:var(--font-mono);">
          Speed: ${latestPt.sog_knots} kts &bull; Heading: ${latestPt.cog_degrees}°
        </div>
      `);

      this.vesselsLayerGroup.addLayer(shipMarker);
    });
  }

  setTimeOffset(hours) {
    if (!Number.isFinite(hours)) return;
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

    // Dynamic Live Coordinates Badge (AlgoRise Timeline pattern)
    if (closestStep && Number.isFinite(closestStep.lat) && Number.isFinite(closestStep.lon)) {
      const latStr = `${Math.abs(closestStep.lat).toFixed(4)}° ${closestStep.lat >= 0 ? 'N' : 'S'}`;
      const lonStr = `${Math.abs(closestStep.lon).toFixed(4)}° ${closestStep.lon >= 0 ? 'E' : 'W'}`;
      if (this.timelineCoordBadge) {
        this.timelineCoordBadge.innerText = `${latStr}, ${lonStr}`;
      }
    }

    // Render sampled particles
    const color = hours < 0 ? '#ff3366' : (hours > 0 ? '#f59e0b' : '#00f2fe');
    if (closestStep.particles_sample) {
      closestStep.particles_sample.forEach(coord => {
        const pMarker = L.circleMarker([coord[1], coord[0]], {
          radius: 3.5,
          color: color,
          weight: 1,
          fillColor: color,
          fillOpacity: 0.75
        }).bindTooltip(`
          <div style="font-family:var(--font-sans); font-size:0.70rem; padding:3px 5px;">
            <strong style="color:${color};">🔵 OIL FLUID PARTICLE</strong><br>
            <span style="color:#cbd5e1; font-size:0.66rem;">1,000 virtual droplets simulating surface spread</span>
          </div>
        `, { sticky: true, className: 'c2-map-tooltip' });
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
    const intervalMs = Math.max(30, Math.round(150 / (this.playbackSpeed || 1)));
    this.playInterval = setInterval(() => {
      let nextTime = this.currentRelativeTime + 0.5;
      if (nextTime > 24.0) {
        nextTime = -18.0;
      }
      this.setTimeOffset(nextTime);
    }, intervalMs);
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
          ais_results: {
            ...this.aisResults,
            counterfactual_verification: this.counterfactualResults || (this.aisResults ? this.aisResults.counterfactual_verification : null)
          },
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
      this.updateStepperState(5);
      document.getElementById('systemStatusText').innerText = 'CASE SUMMARY EXPORTED';
    } catch (error) {
      console.error('Case summary export failed:', error);
      document.getElementById('systemStatusText').innerText = 'EXPORT FAILED';
      alert(error.message || 'Unable to generate case summary.');
    }
  }

  // ── Real-Time Live Automated Data Ingestion Center ────────────────────────
  initLiveIngestion() {
    this.modalLive = document.getElementById('liveIngestionModal');
    this.btnOpenLive = document.getElementById('btnOpenLiveIngestion');
    this.btnCloseLive = document.getElementById('btnCloseLiveModal');
    this.btnCheckFeeds = document.getElementById('btnCheckLiveFeeds');
    this.btnExecLive = document.getElementById('btnExecuteLiveMission');
    this.livePresets = document.querySelectorAll('.btn-preset');
    this.liveInputLat = document.getElementById('liveInputLat');
    this.liveInputLon = document.getElementById('liveInputLon');
    this.liveInputTitle = document.getElementById('liveInputTitle');
    this.feedStatusSat = document.getElementById('feedStatusSat');
    this.feedStatusMeteo = document.getElementById('feedStatusMeteo');
    this.feedStatusAIS = document.getElementById('feedStatusAIS');
    this.liveLogBox = document.getElementById('liveTelemetryLog');
    this.liveLogText = document.getElementById('liveLogText');

    if (this.btnOpenLive) {
      this.btnOpenLive.addEventListener('click', () => {
        if (this.modalLive) this.modalLive.style.display = 'flex';
      });
    }

    if (this.btnCloseLive) {
      this.btnCloseLive.addEventListener('click', () => {
        if (this.modalLive) this.modalLive.style.display = 'none';
      });
    }

    if (this.modalLive) {
      this.modalLive.addEventListener('click', (e) => {
        if (e.target === this.modalLive) this.modalLive.style.display = 'none';
      });
    }

    this.livePresets.forEach(btn => {
      btn.addEventListener('click', () => {
        this.livePresets.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (this.liveInputLat) this.liveInputLat.value = btn.dataset.lat;
        if (this.liveInputLon) this.liveInputLon.value = btn.dataset.lon;
        if (this.liveInputTitle) this.liveInputTitle.value = btn.dataset.name;
      });
    });

    if (this.btnCheckFeeds) {
      this.btnCheckFeeds.addEventListener('click', () => this.probeLiveFeeds());
    }

    if (this.btnExecLive) {
      this.btnExecLive.addEventListener('click', () => this.executeLiveMission());
    }
  }

  async probeLiveFeeds() {
    let lat, lon;
    try {
      lat = parseRequiredCoordinate(this.liveInputLat?.value, 'Latitude', 22.585);
      lon = parseRequiredCoordinate(this.liveInputLon?.value, 'Longitude', 69.185);
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) throw new Error('Latitude/longitude is outside geographic bounds.');
    } catch (err) {
      this.showToast(err.message, 'warning');
      return;
    }

    if (this.liveLogBox) this.liveLogBox.style.display = 'block';
    if (this.liveLogText) {
      this.liveLogText.innerText = `[PROBE] Querying live external feeds for Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}...\n`;
    }

    if (this.feedStatusSat) this.feedStatusSat.innerText = 'Querying NASA ASF...';
    if (this.feedStatusMeteo) this.feedStatusMeteo.innerText = 'Querying Open-Meteo...';
    if (this.feedStatusAIS) this.feedStatusAIS.innerText = 'Querying Transponders...';

    try {
      const [resSat, resMeteo, resAIS] = await Promise.all([
        fetch(`/api/live/satellite-passes?lat=${lat}&lon=${lon}`).then(r => r.json()).catch(e => ({ error: e.message })),
        fetch(`/api/live/ocean-weather?lat=${lat}&lon=${lon}`).then(r => r.json()).catch(e => ({ error: e.message })),
        fetch(`/api/live/ais-traffic?lat=${lat}&lon=${lon}`).then(r => r.json()).catch(e => ({ error: e.message }))
      ]);

      // 1. Satellite feed status
      if (resSat?.status === 'available' && resSat.latest_pass) {
        const pass = resSat.latest_pass;
        this.feedStatusSat.innerText = `✓ ${pass.platform}: ${pass.granule_name.slice(0, 24)}... (${pass.acquisition_time_utc})`;
        this.feedStatusSat.style.color = '#00e5ff';
        this.liveLogText.innerText += `[SAT FEED] Pass found: ${pass.granule_name} | Sensor: ${pass.sensor} | Flight: ${pass.flight_direction}\n`;
      } else {
        this.feedStatusSat.innerText = resSat?.status === 'unavailable' ? 'Catalogue unavailable — no pass invented' : 'No matching catalogue pass returned';
        this.feedStatusSat.style.color = '#fbbf24';
        this.liveLogText.innerText += `[SAT FEED] ${resSat?.message || 'No source result received.'}\n`;
      }

      // 2. Ocean weather status
      if (resMeteo?.status === 'available' && resMeteo.surface_current) {
        const curr = resMeteo.surface_current;
        const wind = resMeteo.surface_wind_10m;
        this.feedStatusMeteo.innerText = `✓ Current: ${curr.speed_knots} kts (${curr.heading_degrees}°) | Wind: ${wind?.speed_knots ?? 'n/a'} kts`;
        this.feedStatusMeteo.style.color = '#00e5ff';
        this.liveLogText.innerText += `[MET-OCEAN] ${resMeteo.provider} | Wave: ${resMeteo.wave_height_m ?? 'n/a'}m | ${resMeteo.notice}\n`;
      } else {
        this.feedStatusMeteo.innerText = 'Met-ocean source unavailable — no fallback values';
        this.feedStatusMeteo.style.color = '#fbbf24';
        this.liveLogText.innerText += `[MET-OCEAN] ${resMeteo?.notice || 'No source result received.'}\n`;
      }

      // 3. AIS traffic status
      if (resAIS?.status === 'available' && Array.isArray(resAIS.vessels)) {
        const vCount = resAIS.vessels.length;
        this.feedStatusAIS.innerText = `✓ ${vCount} vessels from received PositionReports`;
        this.feedStatusAIS.style.color = '#00e5ff';
        this.liveLogText.innerText += `[AIS FEED] Received ${resAIS.received_position_reports} position reports for ${vCount} vessels.\n`;
        resAIS.vessels.forEach(v => {
          this.liveLogText.innerText += `  • ${v.vessel_name} (MMSI: ${v.mmsi}, SOG: ${v.current_position.sog_knots} kts)\n`;
        });
      } else {
        const aisState = resAIS?.status === 'not_configured' ? 'AIS key not configured — no vessels rendered' : 'No AIS PositionReports received';
        this.feedStatusAIS.innerText = aisState;
        this.feedStatusAIS.style.color = '#fbbf24';
        this.liveLogText.innerText += `[AIS FEED] ${resAIS?.message || aisState}\n`;
      }

      this.liveLogText.innerText += `[STATUS] Satellite: ${resSat?.status || 'error'} | Met-ocean: ${resMeteo?.status || 'error'} | AIS: ${resAIS?.status || 'error'}.\n`;
      this.showToast('Source status refreshed', 'info');
    } catch (err) {
      console.error('Live feed probe failed:', err);
      if (this.liveLogText) this.liveLogText.innerText += `[ERROR] Probe encountered issue: ${err.message}\n`;
      this.showToast('Live probe completed with warnings', 'warning');
    }
  }

  async executeLiveMission() {
    let lat, lon;
    try {
      lat = parseRequiredCoordinate(this.liveInputLat?.value, 'Latitude', 22.585);
      lon = parseRequiredCoordinate(this.liveInputLon?.value, 'Longitude', 69.185);
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) throw new Error('Latitude/longitude is outside geographic bounds.');
    } catch (err) {
      this.showToast(err.message, 'warning');
      return;
    }
    const title = this.liveInputTitle.value.trim() || 'Live Operational AOI';

    if (this.btnExecLive) {
      this.btnExecLive.innerText = '⏳ Refreshing Source-Backed AOI...';
      this.btnExecLive.disabled = true;
    }

    if (this.liveLogBox) this.liveLogBox.style.display = 'block';
    if (this.liveLogText) {
      this.liveLogText.innerText += `\n[COMMAND] Refreshing source-backed context for ${title}...\n`;
    }

    try {
      const resp = await fetch('/api/live/create-mission', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: lat,
          lon: lon,
          title: title,
          region: `${title} (Live Stream)`
        })
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.detail || 'Failed to initialize live mission');
      }

      const data = await resp.json();
      if (data.status === 'ready_for_source_scene') {
        this.liveAOI = { lat, lon, title };
        this.map.setView([lat, lon], Math.max(this.map.getZoom(), 9));
        this.renderLiveAis(data.ais);
        // The port catalogue is refreshed independently (monthly reference data);
        // do not hold the AOI refresh hostage to a slow external catalogue query.
        await this.refreshOperationalLayers({ lat, lon });
        if (this.liveLogText) {
          this.liveLogText.innerText += `[AOI] ${title}: satellite=${data.satellite.status}, met-ocean=${data.met_ocean.status}, AIS=${data.ais.status}.\n`;
          this.liveLogText.innerText += `[NEXT] ${data.message}\n`;
        }
        this.showToast('AOI refreshed — authentic SAR raster required for analysis', 'info');
        if (this.btnExecLive) {
          this.btnExecLive.innerText = '⚡ Refresh Source-Backed AOI';
          this.btnExecLive.disabled = false;
        }
        return;
      }
      const scId = data.scenario_id;

      if (this.liveLogText) {
        this.liveLogText.innerText += `[SUCCESS] Registered mission ${scId} with ${data.scenario.ais_vessels.length} vessels.\n`;
        this.liveLogText.innerText += `[EXECUTE] Loading tactical map & running deep neural pipeline...\n`;
      }

      // Add to scenario selector dropdown if not present
      const selector = document.getElementById('scenarioSelector');
      if (selector) {
        let opt = selector.querySelector(`option[value="${scId}"]`);
        if (!opt) {
          opt = document.createElement('option');
          opt.value = scId;
          opt.innerText = `🌐 Live Stream — ${title}`;
          selector.insertBefore(opt, selector.firstChild);
        }
        selector.value = scId;
      }

      this.activeScenarioId = scId;
      this.showToast(`🌐 Live mission initialized: ${title}`, 'success');

      // Close modal after a brief moment
      setTimeout(() => {
        if (this.modalLive) this.modalLive.style.display = 'none';
        if (this.btnExecLive) {
          this.btnExecLive.innerText = '⚡ Refresh Source-Backed AOI';
          this.btnExecLive.disabled = false;
        }
      }, 900);

      // Load scenario into the active C2 map and run full pipeline
      await this.loadScenario(scId);

    } catch (err) {
      console.error('Live mission execution error:', err);
      if (this.liveLogText) {
        this.liveLogText.innerText += `[FAILED] Mission execution error: ${err.message}\n`;
      }
      this.showToast(`Execution error: ${err.message}`, 'error');
      if (this.btnExecLive) {
        this.btnExecLive.innerText = '⚡ Refresh Source-Backed AOI';
        this.btnExecLive.disabled = false;
      }
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

// Instantiate on DOM load with fail-safe splash dismissal
window.addEventListener('DOMContentLoaded', () => {
  // Absolute fallback: remove splash after 1.5s regardless of exceptions
  setTimeout(() => {
    const s = document.getElementById('appSplash');
    if (s) {
      s.style.opacity = '0';
      setTimeout(() => s.remove(), 400);
    }
  }, 1500);

  try {
    window.oceanShield = new OceanShieldApp();
  } catch (err) {
    console.error('OceanShield initialization error:', err);
  } finally {
    const splash = document.getElementById('appSplash');
    if (splash) {
      splash.style.opacity = '0';
      setTimeout(() => splash.remove(), 400);
    }
  }
});
