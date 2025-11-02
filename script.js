/* script.js
   - Fetches live inputs for selected cities (Delhi, Mumbai, Chennai, Kolkata, Patna)
   - Displays one editable card per city with all FEATURES
   - Supports client-side KNN (model_data.json) or API mode (POST /predict)
   - If model_data.json contains scaler_mean & scaler_scale, input will be scaled client-side
*/

const FEATURES = [
    'AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
    'temperature', 'humidity', 'wind_speed', 'precipitation',
    'population_density', 'green_cover_percentage'
];

const CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Patna'];

let modelData = null;
let useClient = true;

/* ----------------- UI helpers ----------------- */
function $(id) { return document.getElementById(id); }
function el(tag, props = {}, children = []) {
    const e = document.createElement(tag);
    for (const k in props) {
        if (k === 'class') e.className = props[k];
        else if (k === 'text') e.textContent = props[k];
        else e.setAttribute(k, props[k]);
    }
    (Array.isArray(children) ? children : [children]).forEach(c => { if (!c) return; e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
}

/* ------------- Build city selection chips -------------- */
function buildCityChips() {
    const container = document.getElementById('cityChips');
    container.innerHTML = '';
    for (const city of CITIES) {
        const chip = el('div', { class: 'chip selected', 'data-city': city }, city);
        chip.addEventListener('click', () => chip.classList.toggle('selected'));
        container.appendChild(chip);
    }
}

function selectedCities() {
    return Array.from(document.querySelectorAll('.chip.selected')).map(c => c.dataset.city);
}

/* ------------- Cards (one per city) ------------------- */
function makeCardForCity(city) {
    const card = el('div', { class: 'card', id: `card_${city}` });
    card.appendChild(el('h3', { text: city }));
    const fieldsGrid = el('div', { class: 'fields-grid' });

    for (const f of FEATURES) {
        const field = el('div', { class: 'field' });
        field.appendChild(el('label', { text: f }));
        const input = el('input', { type: 'number', id: `${city}__${f}`, placeholder: 'leave blank to use model/data default' });
        input.addEventListener('input', () => { /* allow editing */ });
        field.appendChild(input);
        fieldsGrid.appendChild(field);
    }
    card.appendChild(fieldsGrid);

    const actions = el('div', { class: 'card-actions' });
    const predictBtn = el('button', { class: 'primary', text: 'Predict' });
    const editBtn = el('button', { text: 'Toggle edit' });
    const meta = el('div', { class: 'small-muted', id: `meta_${city}`, text: '' });
    predictBtn.addEventListener('click', () => predictForCity(city));
    editBtn.addEventListener('click', () => toggleCardInputs(city));
    actions.appendChild(predictBtn);
    actions.appendChild(editBtn);
    actions.appendChild(meta);
    card.appendChild(actions);

    const resBox = el('div', { class: 'result-box', id: `result_${city}`, text: 'No prediction yet' });
    card.appendChild(resBox);

    return card;
}

function toggleCardInputs(city) {
    const inputs = document.querySelectorAll(`#card_${city} input`);
    inputs.forEach(i => i.readOnly = !i.readOnly);
}

/* ---------- load client-side model_data.json ---------- */
async function loadModelData() {
    try {
        const r = await fetch('model_data.json');
        if (!r.ok) { modelData = null; console.warn('model_data.json not present'); return false; }
        modelData = await r.json();
        console.log('Loaded client model_data.json', modelData);
        // If scaler fields exist, ensure they are arrays of length D
        return true;
    } catch (err) {
        console.warn('Error loading model_data.json', err);
        modelData = null;
        return false;
    }
}

/* ------------- KNN in JS (simple) ---------------- */
function euclidean(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const da = a[i], db = b[i];
        if (Number.isNaN(da) || Number.isNaN(db)) { s += 1e6; continue; } // penalize missing
        const d = da - db; s += d * d;
    }
    return Math.sqrt(s);
}
function knnPredictSingle(x_query, k) {
    const X = modelData.X_train;
    const y = modelData.y_train;
    const N = X.length;
    const arr = new Array(N);
    for (let i = 0; i < N; i++) { arr[i] = { d: euclidean(x_query, X[i]), idx: i }; }
    arr.sort((a, b) => a.d - b.d);
    const top = arr.slice(0, Math.min(k, N));
    let sum = 0; for (const t of top) sum += y[t.idx];
    return sum / top.length;
}

/* ------------- scaling support ----------------- */
function maybeScaleArray(arr) {
    // if modelData contains scaler_mean & scaler_scale, apply: (arr - mean)/scale
    if (!modelData) return arr;
    if (modelData.scaler_mean && modelData.scaler_scale) {
        const mean = modelData.scaler_mean;
        const scale = modelData.scaler_scale;
        const scaled = arr.map((v, i) => Number.isNaN(v) ? NaN : ((v - mean[i]) / (scale[i] === 0 ? 1e-6 : scale[i])));
        return scaled;
    }
    // else, assume X_train already scaled and arr must be scaled externally; we will try to fallback by using means in modelData
    return arr;
}

/* ------------- Predict for one city (client or API) -------------- */
async function predictForCity(city) {
    const mode = document.getElementById('modeSelect').value;
    const apiUrl = document.getElementById('apiUrl').value.trim();

    // read inputs for this city (if blank put NaN)
    const raw = FEATURES.map(f => {
        const v = document.getElementById(`${city}__${f}`).value;
        return v === '' ? NaN : Number(v);
    });

    // if client mode, fill NaNs with column means (if present) and scale if scaler provided
    if (mode === 'client') {
        if (!modelData) { document.getElementById(`result_${city}`).textContent = 'No client model loaded'; return; }
        // compute column means from X_train if provided
        const D = modelData.X_train[0].length;
        const means = new Array(D).fill(0);
        for (let j = 0; j < D; j++) {
            let s = 0, c = 0;
            for (let i = 0; i < modelData.X_train.length; i++) {
                const v = modelData.X_train[i][j];
                if (!Number.isNaN(v)) { s += v; c++; }
            }
            means[j] = c > 0 ? s / c : 0;
        }
        // If modelData indicates X_train is SCALED (recommended) then the inputs must be scaled too.
        let xPrepared = raw.slice();
        // Replace NaNs with either unscaled mean if scaler not provided (rare), else we will replace with mean of original feature
        for (let i = 0; i < xPrepared.length; i++) { if (Number.isNaN(xPrepared[i])) xPrepared[i] = modelData.input_fill && modelData.input_fill[i] !== undefined ? modelData.input_fill[i] : means[i]; }
        // If scaler exists in modelData apply scaling
        if (modelData.scaler_mean && modelData.scaler_scale) {
            xPrepared = xPrepared.map((v, i) => ((v - modelData.scaler_mean[i]) / (modelData.scaler_scale[i] === 0 ? 1e-6 : modelData.scaler_scale[i])));
        } else {
            // If X_train in modelData already scaled, we assume modelData.X_train is scaled and we need to scale inputs similarly.
            // If you saved model_data.json with X_train already scaled and without scaler_mean/scale, you should save scaler or provide scaled inputs.
            // Here we attempt to fall back: if modelData.X_train mean is near 0 and std near 1, assume scaled and we replace NaN with 0.
            xPrepared = xPrepared.map((v, i) => Number.isFinite(v) ? v : 0);
        }

        const k = modelData.k || 5;
        try {
            const pred = knnPredictSingle(xPrepared, Math.min(k, modelData.X_train.length));
            document.getElementById(`result_${city}`).textContent = Number.isFinite(pred) ? pred.toFixed(3) : String(pred);
            document.getElementById(`meta_${city}`).textContent = `client KNN (k=${k}, n_train=${modelData.X_train.length})`;
        } catch (err) {
            document.getElementById(`result_${city}`).textContent = 'Client prediction error: ' + err.message;
        }

    } else {
        // API mode: send raw values (unscaled). Your API should accept {features: [...]} in same order and do any scaling there.
        if (!apiUrl) { document.getElementById(`result_${city}`).textContent = 'Set API URL in config'; return; }
        document.getElementById(`result_${city}`).textContent = 'Predicting...';
        try {
            const resp = await fetch(apiUrl, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: raw, city: city })
            });
            if (!resp.ok) { const txt = await resp.text(); throw new Error(txt || 'API error'); }
            const j = await resp.json();
            if (typeof j.prediction === 'undefined') { throw new Error('Invalid API response'); }
            document.getElementById(`result_${city}`).textContent = Number.isFinite(j.prediction) ? Number(j.prediction).toFixed(3) : j.prediction;
            document.getElementById(`meta_${city}`).textContent = `via API`;
        } catch (err) {
            document.getElementById(`result_${city}`).textContent = 'API error: ' + err.message;
        }
    }
}

/* ------------- Fetch live data for a city (OpenWeatherMap + OpenAQ example) -------------- */
/*
  IMPORTANT: Insert your API keys below.
  - OpenWeatherMap (current weather): temperature, humidity, wind, precipitation
  - OpenAQ: pollutant values (pm25, pm10, no2, o3, so2, co)
*/
const OPENWEATHER_KEY = 'YOUR_OPENWEATHERMAP_API_KEY';
const OPENAQ_BASE = 'https://api.openaq.org/v2/latest'; // free

async function fetchLiveForCity(city) {
    // Map structure we will fill
    const mapping = {
        'AQI': NaN, 'PM2.5': NaN, 'PM10': NaN, 'NO2': NaN, 'SO2': NaN, 'CO': NaN, 'O3': NaN,
        'temperature': NaN, 'humidity': NaN, 'wind_speed': NaN, 'precipitation': NaN,
        'population_density': NaN, 'green_cover_percentage': NaN
    };

    // 1) OpenWeather (city)
    try {
        if (OPENWEATHER_KEY && OPENWEATHER_KEY !== 'a5bbf40c9bbcd1843a7187b0f4a1520a2afbc0348d357d6c936e29d487dba92e') {
            const owUrl = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&units=metric&appid=${OPENWEATHER_KEY}`;
            const owResp = await fetch(owUrl);
            if (owResp.ok) {
                const ow = await owResp.json();
                mapping.temperature = ow?.main?.temp ?? NaN;
                mapping.humidity = ow?.main?.humidity ?? NaN;
                mapping.wind_speed = ow?.wind?.speed ?? NaN;
                // precipitation in mm for last 1 hour if provided
                mapping.precipitation = (ow?.rain?.['1h'] ?? ow?.snow?.['1h']) ?? NaN;
            }
        }
    } catch (err) {
        console.warn('OpenWeather error for', city, err);
    }

    // 2) OpenAQ for pollutant concentrations (free, may not always have city-level records)
    try {
        const oaUrl = `${OPENAQ_BASE}?city=${encodeURIComponent(city)}&limit=100`;
        const qaResp = await fetch(oaUrl);
        if (qaResp.ok) {
            const qa = await qaResp.json();
            // collect measurements
            const measures = {};
            if (Array.isArray(qa.results)) {
                qa.results.forEach(loc => {
                    if (!Array.isArray(loc.measurements)) return;
                    loc.measurements.forEach(m => {
                        const k = (m.parameter || '').toLowerCase();
                        if (!measures[k]) measures[k] = [];
                        measures[k].push(m.value);
                    });
                });
            }
            const avg = k => measures[k] ? (measures[k].reduce((a, b) => a + b, 0) / measures[k].length) : NaN;
            mapping['PM2.5'] = mapping['PM2.5'] || avg('pm25');
            mapping['PM10'] = mapping['PM10'] || avg('pm10');
            mapping['NO2'] = mapping['NO2'] || avg('no2');
            mapping['SO2'] = mapping['SO2'] || avg('so2');
            mapping['O3'] = mapping['O3'] || avg('o3');
            mapping['CO'] = mapping['CO'] || avg('co');
            // AQI not provided by OpenAQ; you could compute AQI locally from concentrations if desired.
        }
    } catch (err) {
        console.warn('OpenAQ error for', city, err);
    }

    // 3) population_density & green_cover - not available from these APIs.
    // You can maintain a small static lookup table (below we try to fetch from modelData if included)
    if (modelData && modelData.city_static && modelData.city_static[city]) {
        const cs = modelData.city_static[city];
        mapping['population_density'] = cs.population_density ?? mapping['population_density'];
        mapping['green_cover_percentage'] = cs.green_cover_percentage ?? mapping['green_cover_percentage'];
    }

    return mapping;
}

/* ------------- Fill the card inputs for a city -------------- */
function setCardInputsFromMapping(city, mapping) {
    for (const f of FEATURES) {
        const elid = `${city}__${f}`;
        const input = document.getElementById(elid);
        if (!input) continue;
        const v = mapping[f];
        input.value = (v === undefined || v === null) ? '' : (Number.isFinite(v) ? Number(v).toFixed(3) : v);
        // Keep inputs read-only so they are displayed but not accidentally edited; user can toggle edit
        input.readOnly = true;
    }
}

/* ------------- Main fetch & display flow for selected cities -------------- */
async function fetchLiveForSelectedCities() {
    const cities = selectedCities();
    if (!cities.length) { alert('Select at least one city'); return; }
    document.getElementById('cardsContainer').innerHTML = '';
    // prepare cards
    for (const city of cities) {
        const c = makeCardForCity(city);
        document.getElementById('cardsContainer').appendChild(c);
    }

    // try to (re)load modelData for client mode
    await loadModelData();

    // fetch live data for each city and populate card
    for (const city of cities) {
        try {
            const mapping = await fetchLiveForCity(city);
            setCardInputsFromMapping(city, mapping);
            // automatically run prediction for each city after filling inputs
            await predictForCity(city);
        } catch (err) {
            console.warn('Error fetching/predicting for', city, err);
            document.getElementById(`result_${city}`).textContent = 'Fetch/predict error: ' + err.message;
        }
    }
}

/* ------------- UI wiring -------------- */
document.addEventListener('DOMContentLoaded', async () => {
    buildCityChips();
    await loadModelData();

    document.getElementById('fetchCitiesBtn').addEventListener('click', fetchLiveForSelectedCities);
    document.getElementById('predictAllBtn').addEventListener('click', async () => {
        const cities = selectedCities();
        for (const city of cities) { await predictForCity(city); }
    });
    document.getElementById('clearBtn').addEventListener('click', () => { document.getElementById('cardsContainer').innerHTML = ''; });

    document.getElementById('modeSelect').addEventListener('change', (e) => {
        const m = e.target.value;
        document.getElementById('apiConfig').classList.toggle('hidden', m === 'client');
    });
});
