/* script.js — rewritten (defensive) */
/* Uses OpenAQ → WeatherAPI → OpenWeather to gather PM2.5/PM10.
   Computes India (CPCB) AQI from PM2.5 & PM10 and writes integer AQI.
   Keep API keys server-side in production. */

const OPENWEATHER_KEY = '4bf0b6858782f0a9d0d62454d7da65f4';
const OPENAQ_BASE = 'https://api.openaq.org/v2/latest';
const WEATHERAPI_KEY = 'a16e02bb1b734caba0895738251611';

const DEFAULT_FEATURES = [
  'AQI','PM2.5','PM10','NO2','SO2','CO','O3',
  'temperature','humidity','wind_speed','precipitation',
  'population_density','green_cover_percentage'
];

const CITIES = ['Delhi','Mumbai','Chennai','Kolkata','Patna'];

let modelData = null;
let FEATURES = DEFAULT_FEATURES.slice();
let sanitizedFeatureIds = {};

// DOM helpers
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
function idForFeature(feature) { return feature.replace(/[^a-z0-9]+/gi, '_'); }
function buildSanitizedMap() { sanitizedFeatureIds = {}; for (const f of FEATURES) sanitizedFeatureIds[f] = idForFeature(f); }

// chips/cards
function renderChips() {
  const container = $('cityChips');
  if (!container) return;
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

// model loader
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

// KNN client (unchanged)
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

// Prediction
async function predictForCity(city) {
  const mode = $('modeSelect') ? $('modeSelect').value : 'client';
  const apiUrl = $('apiUrl') ? $('apiUrl').value.trim() : '';
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

/* ----------------- AQI (India/CPCB) helpers ----------------- */
function clamp(n, a, b){ return Math.min(Math.max(n,a),b); }
function subIndexFromRange(C, C_low, C_high, I_low, I_high) {
  if (!Number.isFinite(C)) return NaN;
  const Cc = clamp(C, C_low, C_high);
  const aqi = ((I_high - I_low) / (C_high - C_low)) * (Cc - C_low) + I_low;
  return Math.round(aqi);
}
const IND_PM25_BREAKPOINTS = [
  {c_low: 0.0,   c_high: 30.0,  i_low: 0,   i_high: 50},
  {c_low: 30.1,  c_high: 60.0,  i_low: 51,  i_high: 100},
  {c_low: 60.1,  c_high: 90.0,  i_low: 101, i_high: 200},
  {c_low: 90.1,  c_high: 120.0, i_low: 201, i_high: 300},
  {c_low: 120.1, c_high: 250.0, i_low: 301, i_high: 400},
  {c_low: 250.1, c_high: 350.0, i_low: 401, i_high: 450},
  {c_low: 350.1, c_high: 500.0, i_low: 451, i_high: 500}
];
const IND_PM10_BREAKPOINTS = [
  {c_low: 0.0,   c_high: 50.0,  i_low: 0,   i_high: 50},
  {c_low: 50.1,  c_high: 100.0, i_low: 51,  i_high: 100},
  {c_low: 100.1, c_high: 250.0, i_low: 101, i_high: 200},
  {c_low: 250.1, c_high: 350.0, i_low: 201, i_high: 300},
  {c_low: 350.1, c_high: 430.0, i_low: 301, i_high: 400},
  {c_low: 430.1, c_high: 500.0, i_low: 401, i_high: 500}
];
function subIndexForPM25(c) {
  if (!Number.isFinite(c)) return NaN;
  for (const bp of IND_PM25_BREAKPOINTS) if (c >= bp.c_low && c <= bp.c_high) return subIndexFromRange(c, bp.c_low,bp.c_high,bp.i_low,bp.i_high);
  if (c > IND_PM25_BREAKPOINTS[IND_PM25_BREAKPOINTS.length-1].c_high) return IND_PM25_BREAKPOINTS[IND_PM25_BREAKPOINTS.length-1].i_high;
  return NaN;
}
function subIndexForPM10(c) {
  if (!Number.isFinite(c)) return NaN;
  for (const bp of IND_PM10_BREAKPOINTS) if (c >= bp.c_low && c <= bp.c_high) return subIndexFromRange(c, bp.c_low,bp.c_high,bp.i_low,bp.i_high);
  if (c > IND_PM10_BREAKPOINTS[IND_PM10_BREAKPOINTS.length-1].c_high) return IND_PM10_BREAKPOINTS[IND_PM10_BREAKPOINTS.length-1].i_high;
  return NaN;
}
function indiaAQICategory(aqi) {
  if (!Number.isFinite(aqi)) return 'Insufficient data';
  if (aqi <= 50) return 'Good';
  if (aqi <= 100) return 'Satisfactory';
  if (aqi <= 200) return 'Moderate';
  if (aqi <= 300) return 'Poor';
  if (aqi <= 400) return 'Very Poor';
  return 'Severe';
}
function computeIndiaAQIFromMapping(mapping){
  const pm25 = Number.isFinite(mapping['PM2.5']) ? mapping['PM2.5'] : (Number.isFinite(mapping.pm25) ? mapping.pm25 : NaN);
  const pm10 = Number.isFinite(mapping['PM10']) ? mapping['PM10'] : (Number.isFinite(mapping.pm10) ? mapping.pm10 : NaN);
  const a25 = Number.isFinite(pm25) ? subIndexForPM25(pm25) : NaN;
  const a10 = Number.isFinite(pm10) ? subIndexForPM10(pm10) : NaN;
  const arr = [];
  if (Number.isFinite(a25)) arr.push(a25);
  if (Number.isFinite(a10)) arr.push(a10);
  if (!arr.length) return NaN;
  return Math.max(...arr);
}

/* ----------------- Live fetch (defensive & ordered) ----------------- */
async function fetchLiveForCity(city) {
  // mapping: keep keys exactly as FEATURES names, plus some helpers (pm25, pm10)
  const mapping = {}; FEATURES.forEach(f => mapping[f]=NaN);

  const debug = (label,obj) => { try { console.log(label, obj); } catch(e) {} };

  // 1) Try OpenAQ first for PM concentrations (best for pollutants)
  try {
    const oaUrl = `${OPENAQ_BASE}?city=${encodeURIComponent(city)}&limit=100`;
    const r = await fetch(oaUrl);
    if (r.ok) {
      const qa = await r.json();
      debug('OpenAQ raw', qa);
      // collect measures by parameter (lowercase)
      const measures = {};
      qa.results?.forEach(loc => { loc.measurements?.forEach(m => {
        const key = (m.parameter || '').toLowerCase();
        if (!measures[key]) measures[key]=[];
        if (typeof m.value === 'number') measures[key].push(m.value);
      });});
      const avg = key => (measures[key] && measures[key].length) ? measures[key].reduce((a,b)=>a+b,0)/measures[key].length : NaN;
      const pm25 = avg('pm25') || avg('pm2.5') || avg('pm_2_5') || NaN;
      const pm10 = avg('pm10') || avg('pm_10') || NaN;
      if (Number.isFinite(pm25)) { mapping['PM2.5'] = pm25; mapping.pm25 = pm25; }
      if (Number.isFinite(pm10)) { mapping['PM10'] = pm10; mapping.pm10 = pm10; }
      // gases (optional)
      if (Number.isFinite(avg('no2'))) mapping['NO2'] = avg('no2');
      if (Number.isFinite(avg('so2'))) mapping['SO2'] = avg('so2');
      if (Number.isFinite(avg('o3'))) mapping['O3'] = avg('o3');
      if (Number.isFinite(avg('co'))) mapping['CO'] = avg('co');
    } else {
      debug('OpenAQ failed status', r.status);
    }
  } catch (err) { console.warn('OpenAQ error', err); }

  // 2) WeatherAPI (supplement for weather + AQ if OpenAQ didn't yield PMs)
  try {
    if (WEATHERAPI_KEY) {
      const wUrl = `https://api.weatherapi.com/v1/current.json?key=${WEATHERAPI_KEY}&q=${encodeURIComponent(city)}&aqi=yes`;
      const r = await fetch(wUrl);
      if (r.ok) {
        const w = await r.json();
        debug('WeatherAPI raw', w);
        // weather
        if (typeof w?.current?.temp_c === 'number') mapping['temperature'] = mapping['temperature'] && Number.isFinite(mapping['temperature']) ? mapping['temperature'] : w.current.temp_c;
        if (typeof w?.current?.humidity === 'number') mapping['humidity'] = mapping['humidity'] && Number.isFinite(mapping['humidity']) ? mapping['humidity'] : w.current.humidity;
        if (typeof w?.current?.wind_kph === 'number') {
          const mps = w.current.wind_kph / 3.6;
          mapping['wind_speed'] = mapping['wind_speed'] && Number.isFinite(mapping['wind_speed']) ? mapping['wind_speed'] : mps;
        }
        if (typeof w?.current?.precip_mm === 'number') mapping['precipitation'] = mapping['precipitation'] && Number.isFinite(mapping['precipitation']) ? mapping['precipitation'] : w.current.precip_mm;
        // AQ object
        const aq = w?.current?.air_quality;
        if (aq && typeof aq === 'object') {
          // WeatherAPI uses pm2_5, pm10 etc
          if (!Number.isFinite(mapping['PM2.5']) && typeof aq.pm2_5 === 'number') { mapping['PM2.5'] = aq.pm2_5; mapping.pm25 = aq.pm2_5; }
          if (!Number.isFinite(mapping['PM10']) && typeof aq.pm10 === 'number') { mapping['PM10'] = aq.pm10; mapping.pm10 = aq.pm10; }
          if (!Number.isFinite(mapping['NO2']) && typeof aq.no2 === 'number') mapping['NO2'] = aq.no2;
          if (!Number.isFinite(mapping['SO2']) && typeof aq.so2 === 'number') mapping['SO2'] = aq.so2;
          if (!Number.isFinite(mapping['O3']) && typeof aq.o3 === 'number') mapping['O3'] = aq.o3;
          if (!Number.isFinite(mapping['CO']) && typeof aq.co === 'number') mapping['CO'] = aq.co;
          // Do not treat WeatherAPI 'us-epa-index' as numeric IND-AQI.
        }
      } else {
        debug('WeatherAPI failed status', r.status);
      }
    }
  } catch (err) { console.warn('WeatherAPI error', err); }

  // 3) OpenWeather (for weather fields only; air pollution endpoint needs lat/lon so we skip it here)
  try {
    if (OPENWEATHER_KEY) {
      const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&units=metric&appid=${OPENWEATHER_KEY}`;
      const r = await fetch(url);
      if (r.ok) {
        const ow = await r.json();
        debug('OpenWeather raw', ow);
        if (!Number.isFinite(mapping['temperature']) && typeof ow?.main?.temp === 'number') mapping['temperature'] = ow.main.temp;
        if (!Number.isFinite(mapping['humidity']) && typeof ow?.main?.humidity === 'number') mapping['humidity'] = ow.main.humidity;
        if (!Number.isFinite(mapping['wind_speed']) && typeof ow?.wind?.speed === 'number') mapping['wind_speed'] = ow.wind.speed;
        const precip = (ow?.rain && typeof ow.rain['1h'] === 'number') ? ow.rain['1h'] : ((ow?.snow && typeof ow.snow['1h'] === 'number') ? ow.snow['1h'] : NaN);
        if (!Number.isFinite(mapping['precipitation']) && Number.isFinite(precip)) mapping['precipitation'] = precip;
      } else {
        debug('OpenWeather failed status', r.status);
      }
    }
  } catch (err) { console.warn('OpenWeather error', err); }

  // 4) modelData static info
  if (modelData && modelData.city_static && modelData.city_static[city]) {
    const cs = modelData.city_static[city];
    if (typeof cs.population_density !== 'undefined') mapping['population_density'] = cs.population_density;
    if (typeof cs.green_cover_percentage !== 'undefined') mapping['green_cover_percentage'] = cs.green_cover_percentage;
  }

  // coerce string numbers to numbers
  for (const k of Object.keys(mapping)) {
    if (typeof mapping[k] === 'string' && mapping[k] !== '') {
      const n = Number(mapping[k]);
      if (!Number.isNaN(n)) mapping[k] = n;
    }
  }

  // Compute IND-AQI using PM2.5 and PM10 (strictly µg/m3). Prefer values we gathered.
  const computedAQI = computeIndiaAQIFromMapping(mapping);
  if (Number.isFinite(computedAQI)) {
    mapping['AQI'] = Math.round(computedAQI);
    mapping['AQI_category'] = indiaAQICategory(mapping['AQI']);
  } else {
    // if no PM data available, attempt to use any index present (WeatherAPI/us-epa-index or OpenWeather main.aqi) to show a label
    // not ideal — will be 'Insufficient data' if nothing
    mapping['AQI'] = NaN;
    mapping['AQI_category'] = 'Insufficient data';
  }

  return mapping;
}

// Live panel helpers
function showLivePanel() {
  const cities = selectedCities();
  if (!cities.length) return alert('Select at least one city (click chips) to show live data for.');
  const city = cities[0];
  updateLivePanel(city);
  const panel = $('liveDataPanel'); if (panel) panel.classList.remove('hidden');
  const lc = $('liveCityName'); if (lc) lc.textContent = city;
  if ($('autoRefreshToggle') && $('autoRefreshToggle').checked) startAutoRefresh(city);
}
async function updateLivePanel(city) {
  try {
    const mapping = await fetchLiveForCity(city);
    const setText = (id, v, int=false) => { const el = $(id); if (!el) return; if (typeof v === 'number' && Number.isFinite(v)) el.textContent = int ? String(Math.round(v)) : v.toFixed(3); else el.textContent = (v === null || typeof v === 'undefined' || Number.isNaN(v)) ? '—' : String(v); };
    setText('live_temperature', mapping['temperature']);
    setText('live_humidity', mapping['humidity'], true);
    setText('live_wind_speed', mapping['wind_speed']);
    setText('live_precipitation', mapping['precipitation']);
    setText('live_PM2_5', mapping['PM2.5']);
    setText('live_PM10', mapping['PM10']);
    setText('live_NO2', mapping['NO2']);
    setText('live_SO2', mapping['SO2']);
    setText('live_O3', mapping['O3']);
    setText('live_CO', mapping['CO']);
    setText('live_population_density', mapping['population_density'], true);
    setText('live_green_cover_percentage', mapping['green_cover_percentage'], true);
    // AQI and category
    const aqiEl = $('live_AQI'); if (aqiEl) aqiEl.textContent = Number.isFinite(mapping['AQI']) ? String(Math.round(mapping['AQI'])) : '—';
    const catEl = $('live_AQI_category'); if (catEl) catEl.textContent = mapping['AQI_category'] ?? '—';
    const upd = $('liveUpdated'); if (upd) upd.textContent = 'Last updated: ' + (new Date()).toLocaleString();
  } catch (err) {
    console.error('Failed to update live panel', err);
    const upd = $('liveUpdated'); if (upd) upd.textContent = 'Last updated: error';
  }
}

// Auto-refresh
let _autoRefreshTimer = null;
function startAutoRefresh(city) {
  stopAutoRefresh();
  const sec = Math.max(10, Number($('refreshInterval').value || 60));
  _autoRefreshTimer = setInterval(() => updateLivePanel(city), sec*1000);
}
function stopAutoRefresh() { if (_autoRefreshTimer) { clearInterval(_autoRefreshTimer); _autoRefreshTimer = null; } }

// High level flows
async function fetchLiveForSelectedCities() {
  const cities = selectedCities();
  if (!cities.length) return alert('Select at least one city');
  $('cardsContainer').innerHTML = '';
  const editable = $('editableByDefault') ? $('editableByDefault').checked : false;
  for (const city of cities) $('cardsContainer').append(makeCardForCity(city,{editableByDefault: editable}));
  await loadModelData();
  for (const city of cities) {
    try {
      const mapping = await fetchLiveForCity(city);
      for (const f of FEATURES) {
        const id = `${city}__${sanitizedFeatureIds[f]}`;
        const inp = document.getElementById(id);
        if (!inp) continue;
        const v = mapping[f];
        if (f === 'AQI') { inp.value = (Number.isFinite(v) ? String(Math.round(v)) : ''); }
        else inp.value = (typeof v === 'number' && Number.isFinite(v)) ? Number(v).toFixed(3) : '';
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
  const editable = $('editableByDefault') ? $('editableByDefault').checked : false;
  for (const city of cities) $('cardsContainer').append(makeCardForCity(city,{editableByDefault: editable}));
}

// DOM wiring
document.addEventListener('DOMContentLoaded', async () => {
  renderChips();
  buildSanitizedMap();
  await loadModelData();
  const fetchBtn = $('fetchCitiesBtn'); if (fetchBtn) fetchBtn.addEventListener('click', fetchLiveForSelectedCities);
  const createBtn = $('createCardsBtn'); if (createBtn) createBtn.addEventListener('click', createCardsForSelectedCities);
  const predictAll = $('predictAllBtn'); if (predictAll) predictAll.addEventListener('click', async ()=> {
    const cards = document.querySelectorAll('#cardsContainer .card');
    for (const card of cards) {
      const city = card.querySelector('h3').textContent;
      await predictForCity(city);
    }
  });
  const clearBtn = $('clearBtn'); if (clearBtn) clearBtn.addEventListener('click', ()=> { $('cardsContainer').innerHTML = ''; });
  const showLive = $('showLiveBtn'); if (showLive) showLive.addEventListener('click', showLivePanel);
  const autoToggle = $('autoRefreshToggle'); if (autoToggle) autoToggle.addEventListener('change', (e)=> {
    if (e.target.checked) {
      const cities = selectedCities(); if (!cities.length) { alert('Select a city first'); e.target.checked=false; return; }
      startAutoRefresh(cities[0]);
    } else stopAutoRefresh();
  });
  const modeSel = $('modeSelect');
  if (modeSel) {
    modeSel.addEventListener('change', (e)=> { const apiCfg = $('apiConfig'); if (apiCfg) apiCfg.classList.toggle('hidden', e.target.value === 'client'); });
    const apiCfg = $('apiConfig'); if (apiCfg) apiCfg.classList.toggle('hidden', modeSel.value === 'client');
  }
});
