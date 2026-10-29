const $ = (id) => document.getElementById(id);
const API_BASE_URL = String(window.GRIDGUARD_API_BASE_URL || '').replace(/\/+$/, '');
let readings = null, busy = false, revision = 0, latest = null;
const history = [];
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function navigate(view) {
  document.querySelectorAll('.nav').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  $('overview-view').hidden = view !== 'overview';
  $('workspace-view').hidden = !['overview','analysis'].includes(view);
  $('history-view').hidden = view !== 'history';
  $('causes-view').hidden = view !== 'causes';
  $('manual-view').hidden = view !== 'manual';
  $('model-view').hidden = view !== 'model';
  $('page-name').textContent = {overview:'Overview',analysis:'Consumer analysis',manual:'Manual check',history:'Analysis history',causes:'Cause finder',model:'Model insights'}[view];
  if (view === 'history') renderHistory();
}
document.querySelectorAll('.nav').forEach(b => b.addEventListener('click', () => navigate(b.dataset.view)));
for (const id of ['new-analysis','hero-start']) $(id).addEventListener('click', () => {navigate('analysis'); $('consumer-id').focus();});
const apiUrl = path => `${API_BASE_URL}${path}`;
async function requestJSON(path, options = {}, timeout = 120000) {
  const response = await fetch(apiUrl(path), {...options, signal: AbortSignal.timeout(timeout)});
  let body;
  try { body = await response.json(); } catch { body = {detail: 'The service returned an invalid response.'}; }
  return {response, body};
}
async function requestPrediction(payload) {
  return requestJSON('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
}
async function checkHealth() {
  try {
    const {response, body} = await requestJSON('/health', {}, 10000);
    const ready = response.ok && body.model_ready === true;
    $('service-status').className = 'live-status ' + (ready ? 'ready' : 'offline');
    $('service-status').replaceChildren(document.createElement('i'), document.createTextNode(ready ? 'Model online' : 'Model unavailable'));
    if (ready) loadModelInfo();
  } catch (error) {
    $('service-status').className = 'live-status offline';
    $('service-status').replaceChildren(document.createElement('i'), document.createTextNode(error.name === 'TimeoutError' ? 'Service timed out' : 'Backend unavailable'));
  }
}
async function loadModelInfo() {
  try {
    const {response, body} = await requestJSON('/model-info', {}, 10000);
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Model information is unavailable.');
    $('model-readings').textContent = body.expected_readings;
    $('model-ensemble').textContent = body.cnn_ensemble_size;
    $('model-status').textContent = `${body.model_version} · ${body.explainability.length ? body.explainability.join(', ') + ' available' : 'SHAP unavailable'}`;
    $('model-capabilities').textContent = body.prediction_capabilities.join(' · ');
  } catch (error) {
    $('model-status').textContent = error.message;
    $('model-capabilities').textContent = '';
  }
}
function chart(data, svg = $('consumption-chart')) {
  const gradient = 'fill-' + svg.id;
  let content = '<defs><linearGradient id="' + gradient + '" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#399b7a" stop-opacity=".19"/><stop offset="100%" stop-color="#399b7a" stop-opacity="0"/></linearGradient></defs>';
  for(let y=15;y<=215;y+=50) content += `<line x1="0" y1="${y}" x2="800" y2="${y}" stroke="#eef1eb" stroke-dasharray="3 5"/>`;
  if(data) {
    const max = Math.max(1,...data.filter(v => v !== null));
    let segments = [], current = [];
    data.forEach((v,i) => { if(v === null) {if(current.length) segments.push(current);current=[];} else current.push([i/Math.max(data.length-1,1)*800,215-v/max*195]); });
    if(current.length) segments.push(current);
    segments.forEach(points => {
      if (points.length === 1) {content += `<circle cx="${points[0][0].toFixed(2)}" cy="${points[0][1].toFixed(2)}" r="3" fill="#398c72"/>`; return;}
      const d = points.map(([x,y],i) => `${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
      content += `<path d="${d} L${points.at(-1)[0]},215 L${points[0][0]},215 Z" fill="url(#${gradient})"/><path d="${d}" fill="none" stroke="#398c72" stroke-width="1.6" stroke-linejoin="round"/>`;
    });
  }
  svg.innerHTML = content;
}
function setData(values, source, name) {
  if(!Array.isArray(values) || values.length !== 1034) throw new Error('Please provide exactly 1,034 daily readings.');
  if(values.some(v => v !== null && (typeof v !== 'number' || !Number.isFinite(v) || v < 0))) throw new Error('Readings must be non-negative finite numbers or null for missing days.');
  if(values.every(v => v === null)) throw new Error('Please include at least one observed reading.');
  readings = values; revision++; latest = null; $('result').hidden=true;
  const observed = values.filter(v => v !== null);
  $('average').textContent = (observed.reduce((a,b)=>a+b,0)/observed.length).toFixed(2);
  $('reading-count').textContent = '1,034 days';
  $('missing-count').textContent = ((1034-observed.length)/1034*100).toFixed(1)+'%';
  $('peak').textContent = Math.max(...observed).toFixed(1)+' kWh';
  $('source').textContent = source;
  $('data-tag').textContent = source === 'Synthetic sample' ? 'Synthetic sample' : 'Uploaded data';
  $('file-status').textContent = name;
  $('chart-empty').hidden=true; $('input-error').textContent=''; $('analyze').disabled=busy; chart(values);
}
$('load-sample').addEventListener('click', () => {
  const sample = Array.from({length:1034},(_,i)=> i%79 === 0 ? null : Number(Math.max(0,16+5*Math.sin(i/53)+3*Math.sin(i*2*Math.PI/7)+2*Math.cos(i*1.7)-(i>650?7:0)).toFixed(2)));
  if(!$('consumer-id').value) $('consumer-id').value='DEMO_001';
  setData(sample,'Synthetic sample','Presentation sample · 1,034 synthetic daily readings');
});
async function loadFile(file) {
  if(!file) return;
  try {
    if(file.size > 2*1024*1024) throw new Error('Please use a file smaller than 2 MB.');
    const text=(await file.text()).replace(/^\uFEFF/,'').trim();
    let values, id;
    if(text.startsWith('[')||text.startsWith('{')) {
      const parsed=JSON.parse(text); values=Array.isArray(parsed)?parsed:parsed.readings; id=parsed.consumer_id;
    } else {
      let tokens=text.split(/\r\n|[\r\n,]/).map(s=>s.trim());
      if(tokens.at(-1)==='') tokens.pop();
      if(/^(reading|readings|consumption|kwh)$/i.test(tokens[0])) tokens.shift();
      values=tokens.map(s => /^(null|na|n\/a)?$/i.test(s) ? null : Number(s));
    }
    setData(values,'Uploaded file',file.name);
    if(typeof id === 'string') $('consumer-id').value=id.slice(0,200);
  } catch(error) {
    readings=null; revision++; latest=null;
    $('analyze').disabled=true; $('result').hidden=true;
    $('file-status').textContent='Upload rejected: '+file.name;
    $('data-tag').textContent='Invalid upload';
    $('average').textContent='—';
    for(const id of ['reading-count','missing-count','peak','source']) $(id).textContent='—';
    $('chart-empty').hidden=false; chart(null);
    $('input-error').textContent=error.message;
  } finally { $('data-file').value=''; }
}
$('data-file').addEventListener('change', e=>loadFile(e.target.files[0]));
const zone=$('dropzone');
['dragenter','dragover'].forEach(type=>zone.addEventListener(type,e=>{e.preventDefault();zone.classList.add('drag');}));
['dragleave','drop'].forEach(type=>zone.addEventListener(type,e=>{e.preventDefault();zone.classList.remove('drag');}));
zone.addEventListener('drop',e=>loadFile(e.dataTransfer.files[0]));
function download(result) {
  const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), a=document.createElement('a');a.href=url;a.download='gridguard-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderResult(result) {
  const flagged=result.predicted_class===1;
  $('result').innerHTML=`<div class="result-top"><div><p class="eyebrow">ANALYSIS COMPLETE · ${escapeHTML(result.consumer_id || 'UNNAMED CONSUMER')}</p><h2>${flagged?'Flagged for review':'Normal consumption classification'}</h2><p>${escapeHTML(result.interpretation)}</p></div><div><div class="result-score ${flagged?'flagged':''}">${(result.theft_probability*100).toFixed(1)}<small>%</small></div><span class="muted small">Model risk score</span></div></div><div class="result-details"><div><span>CNN score </span>${(result.cnn_prediction*100).toFixed(1)}%</div><div><span>Threshold </span>${(result.threshold*100).toFixed(0)}%</div><div><span>Missing readings </span>${result.missing_pct.toFixed(2)}%</div></div><p>This score is not a calibrated probability or proof of theft. ${result.source==='Synthetic sample'?'Input: synthetic presentation sample. ':''}Human review is required for any flagged finding.</p><div class="manual-actions"><button class="secondary" id="download-result">↓ Download report</button><button class="secondary" id="explain-result">Explain with SHAP</button></div><div id="explanation-result" class="explanation" hidden></div>`;
  $('download-result').addEventListener('click',()=>download(result));
  $('explain-result').addEventListener('click', () => explainResult(result, $('explanation-result'), $('explain-result')));
  $('result').hidden=false;
}
async function explainResult(result, output, button) {
  if (!readings) return;
  button.disabled = true;
  button.textContent = 'Explaining…';
  output.hidden = false;
  output.textContent = '';
  try {
    const {response, body} = await requestJSON('/explain', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({consumer_id: result.consumer_id || null, readings: [...readings]})});
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'SHAP explanation is unavailable.');
    const items = Array.isArray(body.features) ? body.features : [];
    if (!items.length) throw new Error('The service returned no explanation features.');
    output.innerHTML = `<p><strong>SHAP explanation</strong> · ${escapeHTML(body.output_space)} output space. Positive values increase the model margin; negative values decrease it.</p><div class="history-wrap"><table><thead><tr><th>Feature</th><th>Value</th><th>SHAP value</th></tr></thead><tbody>${items.map(item => `<tr><td>${escapeHTML(item.feature)}</td><td>${Number(item.value).toFixed(4)}</td><td>${Number(item.shap_value).toFixed(4)}</td></tr>`).join('')}</tbody></table></div><p class="muted small">Base value: ${Number(body.base_value).toFixed(4)} · Raw margin: ${Number(body.raw_prediction).toFixed(4)} · Prediction: ${(Number(body.prediction) * 100).toFixed(2)}%</p>`;
  } catch (error) {
    output.textContent = error.name === 'TimeoutError' ? 'Explanation timed out. Check the service and try again.' : error.message;
  } finally {
    button.disabled = false;
    button.textContent = 'Explain with SHAP';
  }
}
$('consumer-id').addEventListener('input',()=>{revision++;latest=null;$('result').hidden=true;});
$('analyze').addEventListener('click',async()=> {
  if(!readings||busy)return;
  busy=true; const activeRevision=revision;
  const payload={consumer_id:$('consumer-id').value.trim()||null,readings:[...readings]};
  const source=$('source').textContent;
  $('analyze').disabled=true; $('analyze').textContent='Analyzing consumption…';$('input-error').textContent='';$('result').hidden=true;
  try {
    const {response,body}=await requestPrediction(payload);
    if(!response.ok) throw new Error(typeof body.detail==='string'?body.detail:body.detail?.[0]?.msg||'The model could not process these readings.');
    const result={...body,source,analyzed_at:new Date().toISOString()};
    history.unshift(result);$('history-count').textContent=history.length;
    $('session-total').innerHTML=`${history.length} <small>consumers</small>`;
    $('session-note').textContent=`${history.filter(r=>r.predicted_class===1).length} flagged for review this session`;
    if(activeRevision===revision){latest=result;renderResult(result);$('result').scrollIntoView({behavior:'smooth',block:'nearest'});}
    checkHealth();
  } catch(error) {$('input-error').textContent=error.name==='TimeoutError'?'Analysis timed out. Check the service and try again.':error.message;}
  finally {busy=false;$('analyze').disabled=!readings;$('analyze').innerHTML='Run detection <span>→</span>';}
});
function renderHistory() {
  if(!history.length){$('history-content').innerHTML='<div class="empty-history"><h2>A fresh start.</h2><p>Run a consumer analysis to see your results here.</p></div>';return;}
  $('history-content').innerHTML='<div class="history-wrap"><table><thead><tr><th>Consumer</th><th>Classification</th><th>Risk score</th><th>Source</th><th>Report</th></tr></thead><tbody>'+history.map((r,i)=>`<tr><td>${escapeHTML(r.consumer_id||'Unnamed')}</td><td>${r.predicted_class===1?'Flagged for review':'Normal'}</td><td>${(r.theft_probability*100).toFixed(1)}%</td><td>${escapeHTML(r.source)}</td><td><button class="secondary" data-download="${i}">Download ↓</button></td></tr>`).join('')+'</tbody></table></div>';
  document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>download(history[Number(b.dataset.download)])));
}

const causeEntries = new Map();
const DAY_MS = 86400000;
const parseDay = (text) => /^\d{4}-\d{2}-\d{2}$/.test(text) && !Number.isNaN(Date.parse(text + 'T00:00:00Z')) ? Date.parse(text + 'T00:00:00Z') : null;
const formatDay = (time) => new Date(time).toISOString().slice(0, 10);
const avg = (list) => list.length ? list.reduce((a, b) => a + b, 0) / list.length : 0;
const sortedCauseEntries = () => [...causeEntries.entries()].sort((a, b) => a[0] - b[0]).map(([time, value]) => ({time, value}));
function causeChart(entries) {
  const svg = $('cause-chart');
  let content = '';
  for (let y = 15; y <= 215; y += 50) content += `<line x1="0" y1="${y}" x2="800" y2="${y}" stroke="#eef1eb" stroke-dasharray="3 5"/>`;
  if (entries.length) {
    const max = Math.max(1, ...entries.map(e => e.value));
    const step = entries.length > 1 ? 780 / (entries.length - 1) : 0;
    const points = entries.map((e, i) => [10 + i * step, 215 - e.value / max * 195]);
    const d = points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    content += `<path d="${d}" fill="none" stroke="#398c72" stroke-width="1.6" stroke-linejoin="round"/>`;
    if (entries.length <= 60) content += points.map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" fill="#398c72"/>`).join('');
  }
  svg.innerHTML = content;
  $('cause-chart-empty').hidden = entries.length > 0;
}
function renderCauseEntries() {
  const entries = sortedCauseEntries();
  $('cause-entry-count').textContent = `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`;
  $('cause-analyze').disabled = entries.length < 14;
  $('cause-result').hidden = true;
  causeChart(entries);
  $('cause-table').innerHTML = entries.length
    ? '<table><thead><tr><th>Date</th><th>kWh</th><th></th></tr></thead><tbody>' + entries.map(e => `<tr><td>${formatDay(e.time)}</td><td>${e.value}</td><td><button data-remove="${e.time}" aria-label="Remove ${formatDay(e.time)}">✕</button></td></tr>`).join('') + '</tbody></table>'
    : '';
  document.querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => {causeEntries.delete(Number(b.dataset.remove)); renderCauseEntries();}));
}
function addCauseEntry(dateText, rawValue) {
  const time = parseDay(dateText);
  const value = typeof rawValue === 'number' ? rawValue : Number(String(rawValue).trim());
  if (time === null) throw new Error('Please enter a valid date.');
  if (String(rawValue).trim() === '' || !Number.isFinite(value) || value < 0) throw new Error('Reading must be a non-negative number.');
  causeEntries.set(time, value);
}
$('cause-add').addEventListener('click', () => {
  try {
    addCauseEntry($('cause-date').value, $('cause-value').value);
    const next = parseDay($('cause-date').value) + DAY_MS;
    $('cause-date').value = formatDay(next);
    $('cause-value').value = '';
    $('cause-value').focus();
    $('cause-error').textContent = '';
    renderCauseEntries();
  } catch (error) {$('cause-error').textContent = error.message;}
});
$('cause-value').addEventListener('keydown', e => {if (e.key === 'Enter') $('cause-add').click();});
$('cause-bulk-add').addEventListener('click', () => {
  const lines = $('cause-bulk').value.split(/\r\n|[\r\n]/).map(l => l.trim()).filter(Boolean);
  let added = 0, skipped = 0;
  lines.forEach(line => {
    const [dateText, valueText] = line.split(/[,\t;]|\s+/).map(s => s.trim()).filter(Boolean);
    try {addCauseEntry(dateText, valueText ?? ''); added++;} catch {skipped++;}
  });
  $('cause-error').textContent = skipped ? `${skipped} line(s) were skipped. Use the format 2026-06-01,17.4` : '';
  if (added) $('cause-bulk').value = '';
  renderCauseEntries();
});
$('cause-sample').addEventListener('click', () => {
  causeEntries.clear();
  const start = Date.parse('2026-06-01T00:00:00Z');
  for (let i = 0; i < 90; i++) {
    const weekday = new Date(start + i * DAY_MS).getUTCDay();
    let value = 20 + 2.5 * Math.sin(i / 9) + (weekday === 5 || weekday === 6 ? 2 : 0) + 1.2 * Math.cos(i * 1.9);
    if (i >= 52) value *= 0.58;
    if (i === 20) value *= 2.6;
    causeEntries.set(start + i * DAY_MS, Number(Math.max(0, value).toFixed(2)));
  }
  $('cause-error').textContent = '';
  renderCauseEntries();
});
$('cause-clear').addEventListener('click', () => {causeEntries.clear(); $('cause-error').textContent = ''; renderCauseEntries();});
function analyzeCauses(entries) {
  const values = entries.map(e => e.value);
  const n = values.length;
  const overall = avg(values);
  const sd = Math.sqrt(avg(values.map(v => (v - overall) ** 2)));
  const median = [...values].sort((a, b) => a - b)[Math.floor(n / 2)];
  const causes = [];
  const add = (name, score, detail) => causes.push({name, score: Math.min(score, 0.95), detail});
  let best = null;
  for (let k = 5; k <= n - 5; k++) {
    const before = avg(values.slice(0, k)), after = avg(values.slice(k));
    if (before > 0) {
      const change = (after - before) / before;
      if (!best || Math.abs(change) > Math.abs(best.change)) best = {k, change, before, after};
    }
  }
  if (best && best.change <= -0.3) {
    const pct = Math.round(-best.change * 100);
    add('Possible meter tampering or bypass', 0.45 + Math.min(-best.change, 0.5) * 0.8, `Average usage fell by ${pct}% from ${best.before.toFixed(1)} to ${best.after.toFixed(1)} kWh/day starting around ${formatDay(entries[best.k].time)} and stayed low. A sustained drop with no outage is a common signature of tampering, so it deserves a field check.`);
    add('Reduced occupancy or appliance removed', 0.35 + Math.min(-best.change, 0.5) * 0.5, `The same ${pct}% sustained drop can also come from fewer people living there, a switched-off heavy appliance, or solar being installed.`);
  }
  if (best && best.change >= 0.3) {
    const pct = Math.round(best.change * 100);
    add('New heavy appliance or added load', 0.45 + Math.min(best.change, 0.5) * 0.8, `Average usage rose by ${pct}% from ${best.before.toFixed(1)} to ${best.after.toFixed(1)} kWh/day around ${formatDay(entries[best.k].time)} and stayed high, which fits a new air conditioner, heater, or extra occupants.`);
  }
  let run = 0, longest = 0, runEnd = 0;
  values.forEach((v, i) => {
    if (median > 0 && v <= median * 0.1) {run++; if (run > longest) {longest = run; runEnd = i;}} else run = 0;
  });
  if (longest >= 3) add('Vacancy, power outage, or meter fault', 0.5 + longest * 0.04, `${longest} consecutive days were close to zero (ending ${formatDay(entries[runEnd].time)}), far below the usual ${median.toFixed(1)} kWh/day.`);
  const spikes = entries.filter(e => sd > 0 && e.value > overall + 2.5 * sd);
  if (spikes.length) add('One-off high-usage event or reading error', 0.4 + Math.min(spikes.length, 3) * 0.05, `${spikes.length} reading(s) were unusually high, for example ${spikes[0].value} kWh on ${formatDay(spikes[0].time)} versus an average of ${overall.toFixed(1)}. Isolated spikes usually come from events, repairs, or typing errors.`);
  const span = Math.round((entries[n - 1].time - entries[0].time) / DAY_MS) + 1;
  const missing = span - n;
  if (missing / span >= 0.15) add('Meter communication problem or estimated billing', 0.4 + Math.min(missing / span, 0.5) * 0.6, `${missing} of ${span} days in this period have no reading (${Math.round(missing / span * 100)}%), so gaps may hide the real pattern.`);
  const byMonth = {};
  entries.forEach(e => {const m = new Date(e.time).getUTCMonth() + 1; (byMonth[m] ||= []).push(e.value);});
  const months = Object.entries(byMonth).map(([m, list]) => ({m: Number(m), mean: avg(list), count: list.length})).filter(x => x.count >= 5);
  if (months.length >= 2) {
    const hi = months.reduce((a, b) => b.mean > a.mean ? b : a), lo = months.reduce((a, b) => b.mean < a.mean ? b : a);
    if (lo.mean > 0 && hi.mean / lo.mean >= 1.25 && [5, 6, 7, 8, 9].includes(hi.m)) add('Seasonal cooling demand', 0.5 + Math.min(hi.mean / lo.mean - 1.25, 0.4), `Usage in month ${hi.m} averages ${hi.mean.toFixed(1)} kWh/day versus ${lo.mean.toFixed(1)} in month ${lo.m}, which matches higher air-conditioning use in hot months.`);
    if (lo.mean > 0 && hi.mean / lo.mean >= 1.25 && [11, 12, 1, 2].includes(hi.m)) add('Seasonal heating or lighting demand', 0.5 + Math.min(hi.mean / lo.mean - 1.25, 0.4), `Usage in month ${hi.m} averages ${hi.mean.toFixed(1)} kWh/day versus ${lo.mean.toFixed(1)} in month ${lo.m}, which matches heating and longer lighting hours in cold months.`);
  }
  const weekend = entries.filter(e => [5, 6].includes(new Date(e.time).getUTCDay())).map(e => e.value);
  const weekday = entries.filter(e => ![5, 6].includes(new Date(e.time).getUTCDay())).map(e => e.value);
  if (weekend.length >= 4 && weekday.length >= 8 && avg(weekday) > 0) {
    const ratio = avg(weekend) / avg(weekday);
    if (ratio >= 1.2) add('Household pattern: heavier use on weekends', 0.35 + Math.min(ratio - 1.2, 0.4), `Fridays and Saturdays average ${avg(weekend).toFixed(1)} kWh versus ${avg(weekday).toFixed(1)} on other days, typical of a home with people around on days off.`);
    if (ratio <= 0.8) add('Business pattern: heavier use on working days', 0.35 + Math.min(0.8 - ratio, 0.4), `Working days average ${avg(weekday).toFixed(1)} kWh versus ${avg(weekend).toFixed(1)} on Fridays and Saturdays, typical of a shop or office.`);
  }
  if (!causes.length) add('Stable, normal consumption', 0.8, 'No sudden shifts, gaps, spikes, or seasonal swings stand out in these readings.');
  return causes.sort((a, b) => b.score - a.score).slice(0, 4);
}
$('cause-analyze').addEventListener('click', () => {
  const entries = sortedCauseEntries();
  if (entries.length < 14) return;
  const level = score => score >= 0.7 ? ['high', 'HIGH'] : score >= 0.5 ? ['medium', 'MEDIUM'] : ['low', 'LOW'];
  const causes = analyzeCauses(entries);
  $('cause-result').innerHTML = `<div class="result-top"><div><p class="eyebrow">CAUSE ANALYSIS · ${entries.length} READINGS · ${formatDay(entries[0].time)} TO ${formatDay(entries.at(-1).time)}</p><h2>${escapeHTML(causes[0].name)}</h2><p>Most likely explanation, ranked from the pattern in your readings.</p></div></div><div class="cause-list">${causes.map(c => {const [cls, label] = level(c.score); return `<div class="cause"><div class="cause-head"><strong>${escapeHTML(c.name)}</strong><span class="likelihood ${cls}">${label} LIKELIHOOD</span></div><p>${escapeHTML(c.detail)}</p></div>`;}).join('')}</div><p class="muted small">These are rule-based hints from a small sample, not proof of theft or of any other cause. Confirm with an on-site inspection. For a model-based theft score, use Consumer analysis with the full 1,034 readings.</p>`;
  $('cause-result').hidden = false;
  $('cause-result').scrollIntoView({behavior: 'smooth', block: 'nearest'});
});
renderCauseEntries();
chart(null);checkHealth();

const TOTAL_DAYS = 1034;
const MIN_MANUAL_READINGS = 5;
function toReading(token) {
  const text = token.trim();
  if (/^(null|na|n\/a|-)?$/i.test(text)) return null;
  const number = Number(text);
  if (!Number.isFinite(number) || number < 0) throw new Error(`"${text}" is not a valid reading. Use non-negative numbers or null.`);
  return number;
}
function parseManual(text, startValue) {
  const lines = text.split(/\r\n|[\r\n]/).map(s => s.trim()).filter(Boolean);
  if (!lines.length) throw new Error('Please enter at least one reading.');
  const datedLine = /^(\d{1,4})[-\/.](\d{1,2})[-\/.](\d{1,4})\s*[,;:\s]\s*(.+)$/;
  if (lines.some(line => datedLine.test(line))) {
    const parsedLines = lines.map(line => {
      const match = line.match(datedLine);
      if (!match) throw new Error(`The line "${line}" should look like: 20-3-2015, 14.5`);
      return {line, a: match[1], b: match[2], c: match[3], value: match[4]};
    });
    const forcesMonthFirst = parsedLines.some(p => p.a.length < 4 && Number(p.b) > 12);
    const forcesDayFirst = parsedLines.some(p => p.a.length < 4 && Number(p.a) > 12);
    if (forcesMonthFirst && forcesDayFirst) throw new Error('The dates mix day-first and month-first formats. Please use one format.');
    const byTime = new Map();
    for (const p of parsedLines) {
      let year, month, day;
      if (p.a.length === 4) [year, month, day] = [p.a, p.b, p.c];
      else if (p.c.length === 4) [year, month, day] = forcesMonthFirst ? [p.c, p.a, p.b] : [p.c, p.b, p.a];
      else throw new Error(`The date in "${p.line}" needs a 4-digit year.`);
      const time = Date.UTC(Number(year), Number(month) - 1, Number(day));
      const label = `${p.a}-${p.b}-${p.c}`;
      const parts = formatDay(time).split('-').map(Number);
      if (parts[0] !== Number(year) || parts[1] !== Number(month) || parts[2] !== Number(day)) throw new Error(`${label} is not a real date.`);
      if (byTime.has(time)) throw new Error(`The date ${formatDay(time)} appears more than once.`);
      byTime.set(time, toReading(p.value));
    }
    const times = [...byTime.keys()].sort((a, b) => a - b);
    const span = (times.at(-1) - times[0]) / DAY_MS + 1;
    if (span > TOTAL_DAYS) throw new Error(`The dates span ${span} days. The maximum is ${TOTAL_DAYS} days.`);
    const values = Array(span).fill(null);
    for (const time of times) values[(time - times[0]) / DAY_MS] = byTime.get(time);
    return {values, startMs: times[0]};
  }
  if (!startValue) throw new Error('Choose a start date, or write a date before each reading.');
  const [year, month, day] = startValue.split('-').map(Number);
  const values = text.split(/[\s,;]+/).filter(Boolean).map(toReading);
  if (values.length > TOTAL_DAYS) throw new Error(`You entered ${values.length} readings. The maximum is ${TOTAL_DAYS}.`);
  return {values, startMs: Date.UTC(year, month - 1, day)};
}
function explainReadings(values, startMs, result) {
  const observed = values.map((value, index) => ({value, index})).filter(item => item.value !== null);
  const numbers = observed.map(item => item.value);
  const mean = avg(numbers);
  const half = Math.floor(numbers.length / 2);
  const firstHalf = avg(numbers.slice(0, half)), secondHalf = avg(numbers.slice(half));
  const reasons = [];
  if (half >= 3 && firstHalf > 0) {
    const change = (firstHalf - secondHalf) / firstHalf;
    if (change >= 0.25) reasons.push({tone: 'risk', text: `Average use fell ${Math.round(change * 100)}% in the second half of the period (${firstHalf.toFixed(1)} to ${secondHalf.toFixed(1)} kWh/day). A sustained drop like this is a typical sign of meter tampering or a bypass.`});
    else if (change <= -0.25) reasons.push({tone: 'info', text: `Average use rose ${Math.round(-change * 100)}% in the second half (${firstHalf.toFixed(1)} to ${secondHalf.toFixed(1)} kWh/day). An increase is not a theft pattern by itself.`});
  }
  if (values.length >= 14) {
    let worst = null;
    for (let i = 7; i <= values.length - 7; i++) {
      const before = values.slice(i - 7, i).filter(v => v !== null);
      const after = values.slice(i, i + 7).filter(v => v !== null);
      if (before.length < 4 || after.length < 4) continue;
      const beforeMean = avg(before), afterMean = avg(after);
      if (beforeMean <= 0) continue;
      const drop = (beforeMean - afterMean) / beforeMean;
      if (!worst || drop > worst.drop) worst = {drop, index: i, beforeMean, afterMean};
    }
    if (worst && worst.drop >= 0.4) reasons.push({tone: 'risk', text: `A sudden drop of ${Math.round(worst.drop * 100)}% appears around ${formatDay(startMs + worst.index * DAY_MS)}: the weekly average went from ${worst.beforeMean.toFixed(1)} to ${worst.afterMean.toFixed(1)} kWh/day.`});
  }
  const nearZero = numbers.filter(v => v < mean * 0.05).length;
  if (mean > 0 && nearZero >= Math.max(3, Math.ceil(numbers.length * 0.05))) reasons.push({tone: 'risk', text: `${nearZero} days show almost no consumption (under 5% of the average). Repeated near-zero days can indicate a bypassed or disconnected meter.`});
  if (half >= 7) {
    const spread = list => {
      const m = avg(list);
      return m > 0 ? Math.sqrt(avg(list.map(v => (v - m) ** 2))) / m : 0;
    };
    const spreadFirst = spread(numbers.slice(0, half)), spreadSecond = spread(numbers.slice(half));
    if (spreadFirst > 0 && spreadSecond / spreadFirst < 0.4) reasons.push({tone: 'risk', text: 'Daily variation became unusually flat in the second half. Real households fluctuate from day to day, so a flat line can mean the readings are not genuine.'});
  }
  const flagged = result.predicted_class === 1;
  const riskCount = reasons.filter(r => r.tone === 'risk').length;
  if (flagged && !riskCount) reasons.unshift({tone: 'risk', text: 'The model flagged subtler pattern changes that are not a single obvious drop. Please review the readings manually.'});
  if (!flagged && riskCount) reasons.push({tone: 'info', text: 'Some irregularities were found, but the overall model score stayed below the review threshold.'});
  if (!flagged && !riskCount) reasons.unshift({tone: 'ok', text: `No sharp drops, near-zero days, or flattening were found. Average consumption is about ${mean.toFixed(1)} kWh/day.`});
  const missing = (values.length - numbers.length) / values.length;
  if (missing >= 0.3) reasons.push({tone: 'info', text: `${Math.round(missing * 100)}% of the days in this period have no reading, which lowers the reliability of the result.`});
  if (numbers.length < 60) reasons.push({tone: 'info', text: `Only ${numbers.length} observed readings were entered. The model was built for 1,034-day profiles, so a short entry gives a less reliable result.`});
  return reasons;
}
function recordResult(result) {
  history.unshift(result);
  $('history-count').textContent = history.length;
  $('session-total').innerHTML = `${history.length} <small>consumers</small>`;
  $('session-note').textContent = `${history.filter(r => r.predicted_class === 1).length} flagged for review this session`;
}
function renderManual(result, values) {
  const flagged = result.predicted_class === 1;
  const reasons = result.reasons.map(r => `<li class="${r.tone}">${escapeHTML(r.text)}</li>`).join('');
  $('manual-output').innerHTML = `<div class="result-top"><div><p class="eyebrow">MANUAL CHECK · ${escapeHTML(result.consumer_id || 'UNNAMED CONSUMER')}</p><h2>${flagged ? 'Likely theft: flagged for review' : 'No theft indicated'}</h2><p>${escapeHTML(result.interpretation)}</p></div><div><div class="result-score ${flagged ? 'flagged' : ''}">${(result.theft_probability * 100).toFixed(1)}<small>%</small></div><span class="muted small">Model risk score</span></div></div><div class="result-details"><div><span>Days entered </span>${values.length}</div><div><span>Observed </span>${result.observed_days}</div><div><span>Threshold </span>${(result.threshold * 100).toFixed(0)}%</div></div><p class="reasons-title">Why this result</p><ul class="reasons">${reasons}</ul><div class="manual-chart"><svg id="manual-chart" viewBox="0 0 800 230" role="img" aria-label="Entered consumption chart"></svg></div><p class="hint">This score is not a calibrated probability or proof of theft. Human review is required for any flagged finding.</p><div class="manual-actions"><button class="secondary" id="manual-download">↓ Download report</button></div>`;
  chart(values, $('manual-chart'));
  $('manual-download').addEventListener('click', () => download(result));
  $('manual-tag').textContent = flagged ? 'Flagged for review' : 'Normal';
  $('manual-empty').hidden = true;
  $('manual-output').hidden = false;
}
$('manual-analyze').addEventListener('click', async () => {
  if (busy) return;
  const errorBox = $('manual-error');
  errorBox.textContent = '';
  let parsed;
  try {
    parsed = parseManual($('manual-readings').value, $('manual-start').value);
    const observedCount = parsed.values.filter(v => v !== null).length;
    if (observedCount < MIN_MANUAL_READINGS) throw new Error(`Please enter at least ${MIN_MANUAL_READINGS} observed readings.`);
  } catch (error) {
    errorBox.textContent = error.message;
    return;
  }
  const padded = [...parsed.values, ...Array(TOTAL_DAYS - parsed.values.length).fill(null)];
  const payload = {consumer_id: $('manual-id').value.trim() || null, readings: padded};
  const button = $('manual-analyze');
  busy = true;
  button.disabled = true;
  button.textContent = 'Analyzing…';
  try {
    const {response, body} = await requestPrediction(payload);
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : body.detail?.[0]?.msg || 'The model could not process these readings.');
    const observedDays = parsed.values.filter(v => v !== null).length;
    const result = {
      ...body,
      source: 'Manual entry',
      start_date: formatDay(parsed.startMs),
      observed_days: observedDays,
      reasons: [],
      analyzed_at: new Date().toISOString()
    };
    result.reasons = explainReadings(parsed.values, parsed.startMs, result);
    recordResult(result);
    renderManual(result, parsed.values);
  } catch (error) {
    errorBox.textContent = error.name === 'TimeoutError' ? 'Analysis timed out. Check the service and try again.' : error.message;
  } finally {
    busy = false;
    button.disabled = false;
    button.innerHTML = 'Check consumer <span>→</span>';
  }
});
