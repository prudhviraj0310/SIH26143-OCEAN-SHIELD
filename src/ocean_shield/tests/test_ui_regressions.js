// No npm dependencies: exercise actual dashboard methods, not text-only checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({
  console, window: {addEventListener() {}},
  document: {getElementById() {return null;}},
  fetch() {throw new Error('No synthetic observation may trigger a request');},
});
const source = fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf8');
vm.runInContext(source, context);
const App = vm.runInContext('OceanShieldApp', context);
const format = vm.runInContext('screenNumber', context);
const parseOptional = vm.runInContext('parseOptionalFiniteNumber', context);
const element = () => ({innerText: '', className: '', style: {}, classList: {add() {}, remove() {}}});
let passed = 0;
function pass(description) {passed += 1; console.log(`PASS: ${description}`);}

function app() {
  const result = Object.create(App.prototype);
  for (const key of ['metricArea', 'metricMass', 'metricElongation', 'metricConfidence', 'metricAge',
                    'labelConfidence', 'execAreaVal', 'srActiveBadge', 'suspectVesselName', 'suspectIMO',
                    'suspectMMSI', 'suspectFlag', 'suspectType', 'suspectScore', 'suspectSummary',
                    'pillAisQuality', 'pillSignalStrong', 'pillSignalWeak', 'barProx', 'barTime',
                    'barSpeed', 'barType', 'scoreProx', 'scoreTime', 'scoreSpeed', 'scoreType',
                    'cfCentroidError', 'cfContainment', 'cfJaccard', 'cfCausalityScore', 'cfVerdictBadge', 'cfExplanation']) {
    result[key] = element();
  }
  result.map = {removeLayer() {}};
  result.showToast = () => {};
  return result;
}

async function main() {
  assert.equal(format(0, 2), '0.00');
  for (const value of [null, undefined, NaN, Infinity, '95']) assert.equal(format(value), 'NOT ASSESSED');
  assert.equal(parseOptional('', 'Current U'), null);
  assert.equal(parseOptional('0', 'Current U'), 0);
  assert.throws(() => parseOptional('not-a-number', 'Current U'), /Current U must be a finite number/);
  assert.throws(() => parseOptional('Infinity', 'Wind U'), /Wind U must be a finite number/);
  pass('zero preserved; absent/invalid numbers never become confidence');

  const sar = app();
  sar.driftResults = {stale: true}; sar.aisResults = {stale: true};
  sar.renderSarResponse({sar_results: {primary_slick: {
    area_km2: null, centroid: null, polygon_geojson: null, elongation: null,
    confidence_score: null, screening_score: 0,
  }}, super_resolution_metadata: {status: 'INTERPOLATED_DISPLAY_PREVIEW'}});
  assert.equal(sar.metricArea.innerText, 'NOT ASSESSED');
  assert.equal(sar.metricConfidence.innerText, '0.0/100 (uncalibrated)');
  assert.equal(sar.driftResults, null); assert.equal(sar.aisResults, null);
  assert.match(sar.srActiveBadge.innerText, /NOT NATIVE RADIOMETRY/);
  pass('quality-gated SAR renders without inventing geometry or confidence');

  sar.renderSarResponse({sar_results: {primary_slick: null}});
  assert.equal(sar.metricConfidence.innerText, 'NOT ASSESSED');
  assert.equal(sar.metricMass.innerText, 'NOT ASSESSED');
  assert.equal(sar.metricAge.innerText, 'NOT ASSESSED');
  assert.equal(sar.execAreaVal.innerText, 'NOT ASSESSED');
  pass('missing feature clears previous detection metrics');

  const reset = app();
  reset.sarResults = {stale: true}; reset.driftResults = {stale: true}; reset.aisResults = {stale: true};
  reset.eoResults = {stale: true}; reset.sarProvenance = {stale: true}; reset.aisProvenance = {stale: true};
  reset.timeSlider = {value: 4}; reset.pausePlayback = () => {};
  reset.particlesLayerGroup = {clearLayers() {}}; reset.vesselsLayerGroup = {clearLayers() {}};
  reset.darkVesselsLayerGroup = {clearLayers() {}}; reset.counterfactualLayerGroup = {clearLayers() {}};
  reset.kdeContoursLayerGroup = {clearLayers() {}}; reset.playIcon = element();
  reset.trafficRiskPills = {querySelectorAll() {return [];}, querySelector() {return null;}};
  reset.resetState();
  assert.equal(reset.sarResults, null); assert.equal(reset.driftResults, null);
  assert.equal(reset.aisResults, null); assert.equal(reset.eoResults, null);
  pass('scenario reset revokes stale upstream and downstream results');

  const lead = app();
  lead.runCounterfactualVerification = () => {};
  lead.selectSuspectVessel({vessel_name: 'Synthetic unknown', lead_priority_score: null, vessel_type: 'Tanker',
    closest_approach: {distance_nm: null}, kinematics: {}, score_breakdown: {},
    spoofing_audit: {has_anomalies: false}, full_trajectory: [],
    abstention_verdict: {is_abstention: true, status: 'UNAVAILABLE', integrity_status: 'NOT_ASSESSED',
                         reason: 'Coverage and gate unavailable', entropy_metrics: {normalized_entropy: null}},
  });
  assert.match(lead.suspectSummary.innerText, /LEAD WITHHELD.*UNAVAILABLE/);
  assert.equal(lead.pillAisQuality.innerText, 'AIS NOT ASSESSED');
  assert.equal(lead.barType.style.width, '0%'); assert.equal(lead.barSpeed.style.width, '0%');
  assert.equal(lead.pillSignalStrong.innerText, '● 0 High proximity/time cues');
  assert.doesNotMatch(lead.suspectSummary.innerText, /Bayesian|Legal Gate|\d+\.\d+%/);
  pass('selection retains unavailable gate; missing audit is not optimal AIS');

  const cf = app();
  cf.renderCounterfactualVerification({status: 'UNAVAILABLE', verification_metrics: null});
  assert.equal(cf.cfJaccard.innerText, 'NOT ASSESSED');
  assert.equal(cf.cfVerdictBadge.innerText, 'UNAVAILABLE');
  cf.renderCounterfactualVerification({verdict: 'CONDITIONAL_SPATIAL_MISMATCH', verification_metrics: {
    centroid_distance_km: 0, predicted_containment_percent: 0, jaccard_index: 0, physical_causality_score: null,
  }});
  assert.equal(cf.cfJaccard.innerText, '0.000');
  assert.equal(cf.cfContainment.innerText, '0.0');
  assert.equal(cf.cfCausalityScore.innerText, 'NOT ESTIMATED');
  pass('null counterfactual metrics render; zero is not hidden');

  const noObservation = app();
  noObservation.sarResults = {primary_slick: {centroid: null, area_km2: null}};
  await noObservation.runCounterfactualVerification({candidate_release_point: {lat: 0, lon: 0}});
  assert.equal(noObservation.counterfactualResults.status, 'NOT_ASSESSED');
  pass('no observation cannot fall back to release point or fake 12 km2');

  context.document.querySelectorAll = () => [];
  const chamber = app();
  chamber.initMacosQuicklook();
  assert.match(chamber.chamberMilestones[1].why, /cannot establish innocence/);
  assert.match(chamber.chamberMilestones[1].legal, /does not prove innocence or establish chain of custody/);
  assert.doesNotMatch(chamber.chamberMilestones[1].math, /Speed_drop_delta = 0.0|Heading_variance < 3.5/);
  assert.match(chamber.chamberMilestones[4].legal, /No IMO admissibility certification/);
  assert.doesNotMatch(JSON.stringify(chamber.chamberMilestones[4]), /-10.2 dB|Admissible satellite imagery/);
  pass('chamber runtime text discloses descriptive context and unverified admissibility/contrast');

  const elements = new Map();
  for (const id of ['systemStatusText', 'vesselCountTag', 'sarCenterLat', 'sarCenterLon',
                    'sarPixelSize', 'sarAcquisitionUtc']) elements.set(id, {...element(), value: ''});
  elements.get('sarCenterLat').value = '0'; elements.get('sarCenterLon').value = '0';
  elements.get('sarPixelSize').value = '10';
  context.document.getElementById = id => elements.get(id) || null;
  const statuses = [];
  const statusElement = elements.get('systemStatusText');
  Object.defineProperty(statusElement, 'innerText', {get() {return statuses.at(-1) || '';},
    set(value) {statuses.push(value);}, configurable: true});

  const uploads = app();
  uploads.inputModeTag = element(); uploads.inputModeCopy = element(); uploads.sourceSummary = element();
  uploads.runAISCorrelation = async () => {};
  const requests = [];
  const aisFixture = {status: 'success', vessels: [{mmsi: 419001234}], vessels_parsed_count: 1,
    provenance: {source_filename: 'historic.csv', freshness_status: 'UNVERIFIED'}};
  context.fetch = async (url, options) => {requests.push({url, options}); return {ok: true, json: async () => aisFixture};};
  await uploads.handleAISFileDirect({name: 'historic.csv'});
  const event = {target: {files: [{name: 'historic.csv'}], value: 'historic.csv'}};
  await uploads.handleAISUpload(event);
  assert.equal(requests.length, 2);
  assert.ok(requests.every(request => request.url === '/api/upload-ais-csv'));
  assert.ok(statuses.every(value => /UPLOADED AIS/.test(value) && /FRESHNESS UNVERIFIED/.test(value)));
  assert.match(elements.get('vesselCountTag').innerText, /UPLOADED VESSELS.*FRESHNESS UNVERIFIED/);
  assert.equal(uploads.inputModeTag.innerText, 'UPLOADED INPUTS');
  assert.match(uploads.inputModeCopy.innerText, /freshness is unverified/);
  assert.match(uploads.sourceSummary.innerText, /Uploaded AIS: historic.csv \(freshness unverified\)/);
  assert.equal(event.target.value, '');
  pass('CSV file/form uploads never label historic records LIVE and disclose unverified freshness');

  const sarFixture = {provenance: {source_filename: 'historic.png', acquisition_time_utc: '2017-01-01T00:00:00Z',
    freshness_status: 'UNVERIFIED', scene_center: {lat: 0, lon: 0}, image_shape_px: {width: 128, height: 128}, pixel_size_m: 10},
    sar_results: {primary_slick: null}, segmentation_overlay_base64: 'local-fixture-overlay',
    super_resolution_base64: null};
  statuses.length = 0; requests.length = 0;
  uploads.srToggle = {checked: false}; uploads.sarPreviewImg = {src: ''};
  uploads.sarUploadForm = {hidden: false}; uploads.modelSelect = {value: 'cfar_edge'};
  context.fetch = async (url, options) => {requests.push({url, options}); return {ok: true, json: async () => sarFixture};};
  await uploads.handleSARFileDirect({name: 'historic.png'});
  await uploads.runUploadedSAR();
  assert.equal(requests.length, 2);
  assert.ok(requests.every(request => request.url === '/api/analyze-sar-upload'));
  assert.ok(statuses.every(value => /UPLOADED SAR/.test(value) && /FRESHNESS UNVERIFIED/.test(value)));
  assert.match(uploads.sourceSummary.innerText, /Uploaded SAR: historic.png \(freshness unverified\)/);
  assert.equal(uploads.sarPreviewImg.src, 'data:image/png;base64,local-fixture-overlay');
  assert.equal(uploads.sarUploadForm.hidden, true);
  pass('SAR file/form uploads remain freshness-unverified even with an acquisition timestamp');
  console.log(`UI_RECEIPT: ${passed} scenarios passed; DOM/map/fetch stand-ins only, no browser-layout claim`);
}

main().catch(error => {console.error(error); process.exitCode = 1;});
