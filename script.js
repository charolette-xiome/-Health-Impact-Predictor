/* script.js — full file with Live Data panel support */

/* ---------- Config ---------- */
const OPENWEATHER_KEY = '4bf0b6858782f0a9d0d62454d7da65f4'; // your key
const OPENAQ_BASE = 'https://api.openaq.org/v2/latest';

const DEFAULT_FEATURES = [
  'AQI','PM2.5','PM10','NO2','SO2','CO','O3',
  'temperature','humidity','wind_speed','precipitation',
  'population_density','green_cover_percentage'
];

const CITIES = ['Delhi','Mumbai','Chennai','Kolkata','Patna'];

let modelData = null;
let FEATURES = DEFAULT_FEATURES.slice();
let sanitizedFeatureIds = {};

// ----- DOM helpers -----
const $ = id => document.getElementById(id);
function el(tag, props={}, children=[]) {
  const e = document.createElement(tag);
  for (const k in props) {
    if (k === 'class') e.className = props[k];
    else if (k === 'text') e.textContent = props[k];
    else e.setAttribute(k, props[k]);
  }
  (Array.isArray(children) ? children : [children]).forEach(c => {
    if (c === null || c === undefined) return;
    e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  });
  return e;
}

// sanitize feature name for element ids
function idForFeature(feature) { return feature.replace(/[^a-z0-9]+/gi, '_'); }
function buildSanitizedMap() { sanitizedFeatureIds = {}; for (const f of FEATURES) sanitizedFeatureIds[f] = idForFeature(f); }

/* ----------------- City chips & cards ----------------- */
function renderChips() {
  const container = $('cityChips');
  container.innerHTML = '';
  for (const city of CITIES) {
    const chip = el('div', {class:'chip selected', 'data-city': city}, city);
    chip.addEventListener('click', () => chip.classList.toggle('selected'));
    container.appendChild(chip);
  }
}
function selectedCities() {
  return Array.from(document.querySelectorAll('.chip.selected')).map(c => c.dataset.city);
}

function makeCardForCity(city, options={}) {
  const editable = !!options.editableByDefault;
  const card = el('div', {class:'card', id:`card_${city}`});
  card.appendChild(el('h3', {text: city}));

  const fieldsGrid = el('div', {class:'fields-grid'});
  for (const f of FEATURES) {
    const fid = `${city}__${sanitizedFeatureIds[f]}`;
    const field = el('div', {class:'field'});
    field.appendChild(el('label', {text: f}));
    const input = el('input', {type:'number', id:fid, placeholder:'', step:'any'});
    input.readOnly = !editable;
    input.addEventListener('dblclick', () => input.value = '');
    field.appendChild(input);
    fieldsGrid.appendChild(field);
  }
  card.appendChild(fieldsGrid);

  const actions = el('div', {class:'card-actions'});
  const predictBtn = el('button', {class:'primary', text:'Predict'});
  const editBtn = el('button', {text: editable ? 'Lock inputs' : 'Unlock inputs'});
  const meta = el('div', {class:'small-muted', id:`meta_${city}`});
  predictBtn.addEventListener('click', () => predictForCity(city));
  editBtn.addEventListener('click', () => toggleCardInputs(city, editBtn));
  actions.append(predictBtn, editBtn, meta);
  card.append(actions);

  card.append(el('div', {class:'result-box', id:`result_${city}`, text:'No prediction yet'}));
  return card;
}
function toggleCardInputs(city, buttonEl=null) {
  const inputs = Array.from(document.querySelectorAll(`#card_${city} input`));
  if (!inputs.length) return;
  const anyReadOnly = inputs.some(i => i.readOnly);
  const lockAll = !anyReadOnly;
  inputs.forEach(i => i.readOnly = lockAll);
  if (buttonEl) buttonEl.textContent = lockAll ? 'Unlock inputs' : 'Lock inputs';
}

/* ----------------- Model loader ----------------- */
async function loadModelData() {
  try {
    const r = await fetch('model_data.json', {cache:'no-cache'});
    if (!r.ok) { modelData = null; console.warn('model_data.json not found'); return false; }
    const j = await r.json();
    modelData = j;
    if (Array.isArray(j.feature_names)) FEATURES = j.feature_names.slice();
    buildSanitizedMap();
    console.log('Loaded model_data.json — features', FEATURES.length, 'n_train', j.X_train ? j.X_train.length : 'unknown');
    return true;
  } catch (err) {
    console.warn('Error loading model_data.json', err);
    modelData = null;
    return false;
  }
}

/* ----------------- KNN client prediction ----------------- */
function euclidean(a,b) {
  let s=0;
  for (let i=0;i<a.length;i++){ const da=a[i], db=b[i]; const d=(da-db)||0; s+=d*d; }
  return Math.sqrt(s);
}
function knnPredictSingle(x_query,k) {
  if (!modelData || !Array.isArray(modelData.X_train) || !Array.isArray(modelData.y_train)) throw new Error('Client model incomplete');
  const X = modelData.X_train; const y = modelData.y_train;
  const arr = X.map((row,i)=>({d:euclidean(x_query,row), idx:i}));
  arr.sort((a,b)=>a.d-b.d);
  const top = arr.slice(0, Math.min(k, X.length));
  let sum=0; for (const t of top) sum+=y[t.idx];
  return sum/top.length;
}
function computeColumnMeans() {
  if (!modelData || !Array.isArray(modelData.X_train) || modelData.X_train.length===0) return null;
  const D = modelData.X_train[0].length; const means = new Array(D).fill(0);
  for (let j=0;j<D;j++){ let s=0,c=0; for (let i=0;i<modelData.X_train.length;i++){ const v=modelData.X_train[i][j]; if (!Number.isNaN(v)){ s+=v; c++; } } means[j] = c>0 ? s/c : 0; }
  return means;
}
function prepareClientInput(raw) {
  const D = raw.length; let x = raw.slice(0,D);
  if (modelData && Array.isArray(modelData.input_fill)) {
    for (let i=0;i<D;i++) if (Number.isNaN(x[i])) { const v=modelData.input_fill[i]; x[i] = (typeof v==='number' && !Number.isNaN(v)) ? v : NaN; }
  }
  const means = computeColumnMeans();
  if (means) for (let i=0;i<D;i++) if (Number.isNaN(x[i])) x[i]=means[i];
  x = x.map(v => Number.isFinite(v)?v:0);
  if (modelData && Array.isArray(modelData.scaler_mean) && Array.isArray(modelData.scaler_scale)) {
    x = x.map((v,i) => { const mu = modelData.scaler_mean[i] ?? 0; const sc = (modelData.scaler_scale[i]===0?1e-6:(modelData.scaler_scale[i] ?? 1)); return (v-mu)/sc; });
  }
  return x;
}

/* ----------------- Prediction for a city ----------------- */
async function predictForCity(city) {
  const mode = $('modeSelect').value;
  const apiUrl = $('apiUrl').value.trim();
  const raw = FEATURES.map(f => { const id=`${city}__${sanitizedFeatureIds[f]}`; const inp=document.getElementById(id); if (!inp) return NaN; const v=inp.value; return v===''?NaN:Number(v); });

  if (mode === 'client') {
    if (!modelData) { $(`result_${city}`).textContent='No client model loaded'; return; }
    if (!Array.isArray(modelData.X_train) || !Array.isArray(modelData.y_train)) { $(`result_${city}`).textContent='Client model incomplete'; return; }
    try {
      const prepared = prepareClientInput(raw);
      const k = modelData.k || 5;
      const pred = knnPredictSingle(prepared, Math.min(k, modelData.X_train.length));
      $(`result_${city}`).textContent = Number.isFinite(pred) ? pred.toFixed(3) : String(pred);
      $(`meta_${city}`).textContent = `client KNN (k=${k}, n_train=${modelData.X_train.length})`;
    } catch (err) {
      console.error(err); $(`result_${city}`).textContent = 'Client prediction error: ' + (err.message || err);
    }
  } else {
    if (!apiUrl) { $(`result_${city}`).textContent = 'Set API URL in config'; return; }
    try {
      $(`result_${city}`).textContent = 'Predicting...';
      const resp = await fetch(apiUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({features: raw, city})});
      if (!resp.ok) { const txt=await resp.text(); throw new Error(txt||`API error ${resp.status}`); }
      const j = await resp.json();
      if (typeof j.prediction === 'undefined') throw new Error('API response missing prediction');
      $(`result_${city}`).textContent = Number.isFinite(j.prediction) ? Number(j.prediction).toFixed(3) : j.prediction;
      $(`meta_${city}`).textContent = 'via API';
    } catch (err) {
      console.error(err); $(`result_${city}`).textContent = 'API error: ' + (err.message || err);
    }
  }
}

/* ----------------- Live data fetch (OWM + OpenAQ) ----------------- */
async function fetchLiveForCity(city) {
  const mapping = {}; FEATURES.forEach(f=>mapping[f]=NaN);
  // OpenWeather current weather
  try {
    if (OPENWEATHER_KEY && OPENWEATHER_KEY !== '') {
      const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&units=metric&appid=${OPENWEATHER_KEY}`;
      const r = await fetch(url);
      if (r.ok) {
        const ow = await r.json();
        mapping['temperature'] = ow?.main?.temp ?? mapping['temperature'];
        mapping['humidity'] = ow?.main?.humidity ?? mapping['humidity'];
        mapping['wind_speed'] = ow?.wind?.speed ?? mapping['wind_speed'];
        const precip = (ow?.rain && typeof ow.rain['1h'] === 'number') ? ow.rain['1h'] : ((ow?.snow && typeof ow.snow['1h'] === 'number') ? ow.snow['1h'] : NaN);
        mapping['precipitation'] = Number.isFinite(precip) ? precip : mapping['precipitation'];
      } else {
        console.warn('OpenWeather failed', r.status);
      }
    }
  } catch (err) { console.warn('OpenWeather error', err); }

  // OpenAQ pollutants
  try {
    const oaUrl = `${OPENAQ_BASE}?city=${encodeURIComponent(city)}&limit=100`;
    const r = await fetch(oaUrl);
    if (r.ok) {
      const qa = await r.json();
      const measures = {};
      qa.results?.forEach(loc => { loc.measurements?.forEach(m => { const k = (m.parameter||'').toLowerCase(); if (!measures[k]) measures[k]=[]; if (typeof m.value==='number') measures[k].push(m.value); }); });
      const avg = key => (measures[key] && measures[key].length) ? measures[key].reduce((a,b)=>a+b,0)/measures[key].length : NaN;
      mapping['PM2.5'] = Number.isFinite(mapping['PM2.5']) ? mapping['PM2.5'] : avg('pm25');
      mapping['PM10'] = Number.isFinite(mapping['PM10']) ? mapping['PM10'] : avg('pm10');
      mapping['NO2'] = Number.isFinite(mapping['NO2']) ? mapping['NO2'] : avg('no2');
      mapping['SO2'] = Number.isFinite(mapping['SO2']) ? mapping['SO2'] : avg('so2');
      mapping['O3']  = Number.isFinite(mapping['O3'])  ? mapping['O3']  : avg('o3');
      mapping['CO']  = Number.isFinite(mapping['CO'])  ? mapping['CO']  : avg('co');
    }
  } catch (err) { console.warn('OpenAQ error', err); }

  // city static info from modelData (if present)
  if (modelData && modelData.city_static && modelData.city_static[city]) {
    const cs = modelData.city_static[city];
    if (typeof cs.population_density !== 'undefined') mapping['population_density'] = cs.population_density;
    if (typeof cs.green_cover_percentage !== 'undefined') mapping['green_cover_percentage'] = cs.green_cover_percentage;
  }

  return mapping;
}

/* ----------------- Live panel helpers ----------------- */
function showLivePanel() {
  const cities = selectedCities();
  if (!cities.length) return alert('Select at least one city (click chips) to show live data for.');
  const city = cities[0]; // show for the first selected city
  updateLivePanel(city);
  $('liveDataPanel').classList.remove('hidden');
  $('liveCityName').textContent = city;
  // start auto-refresh if toggle is on
  if ($('autoRefreshToggle').checked) startAutoRefresh(city);
}

async function updateLivePanel(city) {
  try {
    const mapping = await fetchLiveForCity(city);
    // populate fields (safely)
    const set = (id, v) => { const el = $(id); if (!el) return; el.textContent = (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(3) : (v === null || typeof v === 'undefined' || Number.isNaN(v) ? '—' : String(v)); };
    set('live_temperature', mapping['temperature']);
    set('live_humidity', mapping['humidity']);
    set('live_wind_speed', mapping['wind_speed']);
    set('live_precipitation', mapping['precipitation']);
    set('live_PM2_5', mapping['PM2.5']);
    set('live_PM10', mapping['PM10']);
    set('live_NO2', mapping['NO2']);
    set('live_SO2', mapping['SO2']);
    set('live_O3', mapping['O3']);
    set('live_CO', mapping['CO']);
    set('live_population_density', mapping['population_density']);
    set('live_green_cover_percentage', mapping['green_cover_percentage']);
    $('liveUpdated').textContent = 'Last updated: ' + (new Date()).toLocaleString();
  } catch (err) {
    console.error('Failed to update live panel', err);
    $('liveUpdated').textContent = 'Last updated: error';
  }
}

/* Auto-refresh logic */
let _autoRefreshTimer = null;
function startAutoRefresh(city) {
  stopAutoRefresh();
  const sec = Math.max(10, Number($('refreshInterval').value || 60));
  _autoRefreshTimer = setInterval(() => updateLivePanel(city), sec*1000);
}
function stopAutoRefresh() { if (_autoRefreshTimer) { clearInterval(_autoRefreshTimer); _autoRefreshTimer = null; } }

/* ----------------- High-level flows ----------------- */
async function fetchLiveForSelectedCities() {
  const cities = selectedCities();
  if (!cities.length) return alert('Select at least one city');
  $('cardsContainer').innerHTML = '';
  const editable = $('editableByDefault').checked;
  for (const city of cities) $('cardsContainer').append(makeCardForCity(city,{editableByDefault: editable}));
  await loadModelData();
  for (const city of cities) {
    try {
      const mapping = await fetchLiveForCity(city);
      // fill card inputs using sanitized ids
      for (const f of FEATURES) {
        const id = `${city}__${sanitizedFeatureIds[f]}`;
        const inp = document.getElementById(id);
        if (!inp) continue;
        const v = mapping[f];
        inp.value = (typeof v === 'number' && Number.isFinite(v)) ? Number(v).toFixed(3) : '';
      }
      await predictForCity(city);
    } catch (err) {
      console.warn('Fetch/predict error for', city, err);
      const r = $(`result_${city}`); if (r) r.textContent = 'Fetch/predict error';
    }
  }
}

function createCardsForSelectedCities() {
  const cities = selectedCities();
  if (!cities.length) return alert('Select at least one city');
  $('cardsContainer').innerHTML = '';
  const editable = $('editableByDefault').checked;
  for (const city of cities) $('cardsContainer').append(makeCardForCity(city,{editableByDefault: editable}));
}

/* ----------------- DOM wiring ----------------- */
document.addEventListener('DOMContentLoaded', async () => {
  // build UI
  renderChips();
  buildSanitizedMap();
  await loadModelData();

  $('fetchCitiesBtn').addEventListener('click', fetchLiveForSelectedCities);
  $('createCardsBtn').addEventListener('click', createCardsForSelectedCities);
  $('predictAllBtn').addEventListener('click', async ()=> {
    const cards = document.querySelectorAll('#cardsContainer .card');
    for (const card of cards) {
      const city = card.querySelector('h3').textContent;
      await predictForCity(city);
    }
  });
  $('clearBtn').addEventListener('click', ()=> { $('cardsContainer').innerHTML = ''; });

  // Live panel buttons
  $('showLiveBtn').addEventListener('click', showLivePanel);
  $('autoRefreshToggle').addEventListener('change', (e)=> {
    if (e.target.checked) {
      const cities = selectedCities(); if (!cities.length) { alert('Select a city first'); e.target.checked=false; return; }
      startAutoRefresh(cities[0]);
    } else stopAutoRefresh();
  });

  // hide apiConfig when client mode selected
  $('modeSelect').addEventListener('change', (e)=> { const apiCfg = $('apiConfig'); if (apiCfg) apiCfg.classList.toggle('hidden', e.target.value === 'client'); });
  const apiCfg = $('apiConfig'); if (apiCfg) apiCfg.classList.toggle('hidden', $('modeSelect').value === 'client');
});
