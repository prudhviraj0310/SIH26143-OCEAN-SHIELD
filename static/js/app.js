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
    this.srActiveBadge = document.getElementById('srActiveBadge');
    this.sarPreviewImg = document.getElementById('sarPreviewImg');

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

    // Professional military-grade Esri World Dark Gray Base (No API key, No watermark)
    L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      attribution: 'Esri &bull; GEBCO &bull; NOAA'
    }).addTo(this.map);

    // Subtle reference labels
    L.tileLayer('https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      opacity: 0.75
    }).addTo(this.map);

    this.particlesLayerGroup = L.layerGroup().addTo(this.map);
    this.vesselsLayerGroup = L.layerGroup().addTo(this.map);

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

    this.btnDetectSAR.addEventListener('click', () => this.runSARAnalysis());
    this.btnRunDrift.addEventListener('click', () => this.runDriftSimulation());
    this.btnCorrelateAIS.addEventListener('click', () => this.runAISCorrelation());
    this.btnDownloadDossier.addEventListener('click', () => this.downloadDossier());

    this.srToggle.addEventListener('change', () => this.toggleSuperResolution());

    this.timeSlider.addEventListener('input', (e) => {
      this.setTimeOffset(parseFloat(e.target.value));
    });

    this.btnPlayPause.addEventListener('click', () => this.togglePlayback());
  }

  async loadScenario(scenarioId) {
    this.resetState();
    try {
      const resp = await fetch(`/api/scenario/${scenarioId}`);
      const data = await resp.json();
      this.scenarioData = data.scenario;

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

      // Automatically trigger end-to-end processing
      await this.runSARAnalysis();
      await this.runDriftSimulation();
      await this.runAISCorrelation();

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
    this.particlesLayerGroup.clearLayers();
    this.vesselsLayerGroup.clearLayers();
  }

  toggleSuperResolution(preloadedData) {
    const isSR = this.srToggle.checked;
    this.srActiveBadge.style.display = isSR ? 'block' : 'none';

    if (preloadedData) {
      this.sarPreviewImg.src = isSR ? preloadedData.sr_sar_image_base64 : preloadedData.sar_image_base64;
    }
  }

  async runSARAnalysis() {
    document.getElementById('systemStatusText').innerText = 'ANALYZING SAR SCENE...';
    try {
      const resp = await fetch('/api/analyze-sar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          use_super_resolution: this.srToggle.checked,
          threshold_offset: 20.0
        })
      });
      const data = await resp.json();
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

        this.map.fitBounds(this.slickLayer.getBounds(), { padding: [50, 50] });

        // Add slick label
        L.popup({ autoClose: false, closeOnClick: false, className: 'slick-tactical-popup' })
          .setLatLng([slick.centroid.lat, slick.centroid.lon])
          .setContent(`<b>${slick.slick_id}</b><br>Observed: ${slick.area_km2.toFixed(2)} km² &bull; ${slick.classification}`)
          .addTo(this.map);
      }

      document.getElementById('systemStatusText').innerText = 'SAR DETECTED';
    } catch (err) {
      console.error('Error analyzing SAR:', err);
    }
  }

  async runDriftSimulation() {
    if (!this.sarResults || !this.sarResults.primary_slick) return;
    const slick = this.sarResults.primary_slick;

    document.getElementById('systemStatusText').innerText = 'COMPUTING HYDRODYNAMICS...';
    try {
      const resp = await fetch('/api/simulate-drift', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: this.activeScenarioId,
          slick_lat: slick.centroid.lat,
          slick_lon: slick.centroid.lon,
          slick_age_hours: slick.estimated_age_hours,
          max_lookback_hours: 18.0,
          forecast_hours: 24.0
        })
      });
      const data = await resp.json();
      this.driftResults = data;

      const origin = data.origin_release_point;
      this.metricOriginCoords.innerText = `${origin.lat.toFixed(4)}° N, ${origin.lon.toFixed(4)}° E (T - ${origin.slick_age_hours}h)`;
      this.metricDriftDist.innerText = `${data.total_drift_distance_km.toFixed(1)} km`;

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
        .bindPopup(`<b>ILLEGAL RELEASE EPICENTER (x₀, y₀)</b><br>Time: T - ${origin.slick_age_hours}h<br>Coordinates: ${origin.lat.toFixed(4)}° N, ${origin.lon.toFixed(4)}° E`)
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
      document.getElementById('systemStatusText').innerText = 'HINDCAST CONVERGED';
    } catch (err) {
      console.error('Error running drift simulation:', err);
    }
  }

  async runAISCorrelation() {
    if (!this.driftResults || !this.driftResults.origin_release_point) return;
    const origin = this.driftResults.origin_release_point;

    document.getElementById('systemStatusText').innerText = 'CORRELATING AIS TRAFFIC...';
    try {
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
          hindcast_trajectory: this.driftResults ? this.driftResults.hindcast_trajectory : null
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
        this.suspectSummary.innerText = `Identified with ${culprit.attribution_tier}. Slowed to ${culprit.kinematics.min_speed_near_origin} kts over epicenter with CPA ${culprit.closest_approach.distance_nm} NM.`;

        // Update Anomaly Breakdown
        const b = culprit.score_breakdown;
        this.barProx.style.width = `${b.proximity_score}%`;
        this.scoreProx.innerText = `${b.proximity_score} / 100`;

        this.barTime.style.width = `${b.temporal_score}%`;
        this.scoreTime.innerText = `${b.temporal_score} / 100`;

        this.barSpeed.style.width = `${b.speed_anomaly_score}%`;
        this.scoreSpeed.innerText = `${b.speed_anomaly_score} / 100 (${culprit.kinematics.speed_drop_knots} kts drop)`;

        this.barType.style.width = `${b.vessel_type_score}%`;
        this.scoreType.innerText = `${b.vessel_type_score} / 100 (${culprit.vessel_type})`;
      }

      // Render Table
      this.renderVesselTable(data.ranked_suspects);

      // Plot Vessel Tracks on Map
      this.renderVesselTracks(data.ranked_suspects);

      document.getElementById('systemStatusText').innerText = 'ROGUE VESSEL ATTRIBUTED';
    } catch (err) {
      console.error('Error correlating AIS:', err);
    }
  }

  renderVesselTable(suspects) {
    this.vesselCountTag.innerText = `${suspects.length} VESSELS`;
    if (!suspects.length) {
      this.vesselTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No vessels in corridor.</td></tr>`;
      return;
    }

    let rowsHtml = '';
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
      }).bindPopup(`<b>${v.vessel_name}</b> (${v.vessel_type})<br>Score: ${v.composite_suspect_score}%<br>Speed: ${latestPt.sog_knots} kts &bull; Heading: ${latestPt.cog_degrees}°`);

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

  downloadDossier() {
    window.location.href = `/api/export-dossier/${this.activeScenarioId}`;
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.oceanShield = new OceanShieldApp();
});
