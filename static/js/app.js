/**
 * Lógica del dashboard.
 * Estado: evaluacionId actual (la última activa). Se reentrena con el botón.
 */
const state = {
  evaluacionId: null,
  equipos: [],
  partidos: [],
  metricas: null,
  ranking: [],
  ultimaPrediccion: null,    // para "guardar al track record"
  chartConsenso: null,
};

const $ = (id) => document.getElementById(id);
const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------------------------------------------------------------------------
// Loading overlay
// ---------------------------------------------------------------------------
function setLoading(on, msg = 'Procesando...') {
  $('loading-text').textContent = msg;
  $('loading').classList.toggle('hidden', !on);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
async function init() {
  try {
    setLoading(true, 'Conectando con el API...');
    await api.health();
    $('health').classList.add('text-emerald-400');

    // Cargar/crear evaluación
    const evals = await api.evaluaciones();
    if (evals.length > 0) {
      state.evaluacionId = evals[evals.length - 1].id;
    } else {
      setLoading(true, 'Entrenando modelo por primera vez (1-2 min)...');
      const nuevaEv = await api.crearEvaluacion();
      state.evaluacionId = nuevaEv.id;
    }
    $('eval-id-text').textContent = `#${state.evaluacionId}`;

    setLoading(true, 'Cargando datos del dashboard...');
    await Promise.all([
      cargarEquipos(), cargarPartidos(), cargarMetricas(), cargarRanking(),
      cargarTrackRecord(), cargarTrackPublico(), cargarFeatureImportance(),
    ]);
    renderKpis();
  } catch (err) {
    console.error(err);
    $('health').classList.remove('text-emerald-400');
    $('health').classList.add('text-rose-500');
    alert(`Error al inicializar: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Equipos (dropdowns)
// ---------------------------------------------------------------------------
async function cargarEquipos() {
  const data = await api.equipos();
  state.equipos = data.equipos;
  for (const sel of ['sel-e1', 'sel-e2']) {
    const el = $(sel);
    el.innerHTML = state.equipos.map(e => `<option value="${e}">${e}</option>`).join('');
  }
  $('sel-e1').value = state.equipos[0] || '';
  $('sel-e2').value = state.equipos[1] || '';
}

// ---------------------------------------------------------------------------
// Historial
// ---------------------------------------------------------------------------
async function cargarPartidos() {
  state.partidos = await api.partidos();
  renderHistorial(state.partidos);
}

function renderHistorial(partidos) {
  const tbody = $('tbl-historial');
  tbody.innerHTML = partidos.map(p => `
    <tr class="border-b border-surface-700/50 hover:bg-surface-700/30">
      <td class="py-2 px-2 text-slate-400 font-mono text-xs">${p.fecha || '—'}</td>
      <td class="py-2 px-2 text-xs"><span class="px-2 py-0.5 rounded bg-surface-700/60 text-slate-300">${p.fase || '—'}</span></td>
      <td class="py-2 px-2">${p.equipo1}</td>
      <td class="py-2 px-2 text-center font-mono">${p.goles_e1} – ${p.goles_e2}</td>
      <td class="py-2 px-2">${p.equipo2}</td>
    </tr>
  `).join('');
}

$('filtro-historial').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  renderHistorial(state.partidos.filter(p =>
    p.equipo1.toLowerCase().includes(q) ||
    p.equipo2.toLowerCase().includes(q) ||
    (p.fase || '').toLowerCase().includes(q)
  ));
});

// ---------------------------------------------------------------------------
// Métricas modelos
// ---------------------------------------------------------------------------
async function cargarMetricas() {
  state.metricas = await api.metricas(state.evaluacionId);
  const tbody = $('tbl-metricas');
  const cv = state.metricas.cv;

  tbody.innerHTML = cv.map(m => {
    const accF = m['CV Mean']  != null ? (m['CV Mean']  * 100).toFixed(1) : '—';
    const f1F  = m['F1 Mean']  != null ? (m['F1 Mean']  * 100).toFixed(1) : '—';
    const lastA= m['Acc Last'] != null ? (m['Acc Last'] * 100).toFixed(0) : '—';
    const std  = m['CV Std']   != null ? (m['CV Std']   * 100).toFixed(1) : '—';
    return `
      <tr class="border-b border-surface-700/50">
        <td class="py-2 px-2 font-medium">${m.Model}</td>
        <td class="py-2 px-2 text-right font-mono">${accF}% <span class="text-slate-500 text-xs">±${std}%</span></td>
        <td class="py-2 px-2 text-right font-mono ${f1F !== '—' && parseFloat(f1F) > 60 ? 'text-emerald-400' : ''}">${f1F}%</td>
        <td class="py-2 px-2 text-right font-mono">${lastA}%</td>
      </tr>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Ranking ELO
// ---------------------------------------------------------------------------
async function cargarRanking() {
  const data = await api.elos(state.evaluacionId);
  state.ranking = data.ranking;
  const tbody = $('tbl-elo');
  const maxElo = Math.max(...state.ranking.map(r => r.elo));
  const minElo = Math.min(...state.ranking.map(r => r.elo));
  const rango  = maxElo - minElo || 1;

  tbody.innerHTML = state.ranking.map((r, i) => {
    const pct = ((r.elo - minElo) / rango) * 100;
    const top = i < 5;
    const bot = i >= state.ranking.length - 5;
    const color = top ? 'bg-emerald-500' : bot ? 'bg-rose-500' : 'bg-brand-500';
    return `
      <tr class="border-b border-surface-700/30">
        <td class="py-1.5 px-2 text-slate-500 font-mono text-xs">${i + 1}</td>
        <td class="py-1.5 px-2 ${top ? 'font-semibold' : ''}">${r.equipo}</td>
        <td class="py-1.5 px-2 text-right font-mono text-xs">${r.elo.toFixed(0)}</td>
        <td class="py-1.5 px-2">
          <div class="h-1.5 bg-surface-700 rounded-full overflow-hidden">
            <div class="${color} h-full" style="width:${Math.max(5, pct)}%"></div>
          </div>
        </td>
      </tr>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Track record (test vs real)
// ---------------------------------------------------------------------------
const MODELOS_TRACK = [
  { key: 'Random Forest',       short: 'RF',  id: 'th-rf'  },
  { key: 'Gradient Boosting',   short: 'GB',  id: 'th-gb'  },
  { key: 'Logistic Regression', short: 'LR',  id: 'th-lr'  },
  { key: 'SVM',                 short: 'SVM', id: 'th-svm' },
  { key: 'XGBoost',             short: 'XGB', id: 'th-xgb' },
  { key: 'KNN',                 short: 'KNN', id: 'th-knn' },
];

const colorByAcc = (acc) =>
  acc >= 0.6  ? 'text-emerald-400'
  : acc >= 0.5 ? 'text-amber-400'
  :              'text-rose-400';

async function cargarTrackRecord() {
  const data = await api.prediccionesTest(state.evaluacionId);
  const tbody = $('tbl-track');
  const cell = (real, pred) => {
    const ok = real === pred;
    const cls = ok ? 'text-emerald-400' : 'text-rose-400';
    return `<td class="py-2 px-2 text-center text-xs font-mono ${cls}">${pred?.[0] || '—'}</td>`;
  };
  tbody.innerHTML = data.predicciones.map(r => `
    <tr class="border-b border-surface-700/50">
      <td class="py-2 px-2">${r.Equipo1}</td>
      <td class="py-2 px-2">${r.Equipo2}</td>
      <td class="py-2 px-2 text-center text-xs font-mono font-semibold">${r.Resultado_Real?.[0] || '—'}</td>
      ${cell(r.Resultado_Real, r['Random Forest'])}
      ${cell(r.Resultado_Real, r['Gradient Boosting'])}
      ${cell(r.Resultado_Real, r['Logistic Regression'])}
      ${cell(r.Resultado_Real, r['SVM'])}
      ${cell(r.Resultado_Real, r['XGBoost'])}
      ${cell(r.Resultado_Real, r['KNN'])}
    </tr>
  `).join('');

  const total = data.predicciones.length;
  for (const m of MODELOS_TRACK) {
    const th = $(m.id);
    if (!th) continue;
    if (!total) { th.textContent = m.short; continue; }
    const aciertos = data.predicciones.filter(p => p[m.key] === p.Resultado_Real).length;
    const acc = aciertos / total;
    const pct = Math.round(acc * 100);
    th.innerHTML = `${m.short} <span class="${colorByAcc(acc)} font-mono ml-1">${pct}%</span>`;
  }
}

// ---------------------------------------------------------------------------
// KPIs
// ---------------------------------------------------------------------------
function renderKpis() {
  const partidos = state.partidos.length;
  const equipos = state.equipos.length;
  const cv = state.metricas?.cv || [];
  const mejor = cv.slice().sort((a, b) => (b['F1 Last'] ?? 0) - (a['F1 Last'] ?? 0))[0];
  const cards = [
    { label: 'Partidos en dataset', value: partidos, sub: `${equipos} equipos` },
    { label: 'Mejor modelo (F1 último fold)', value: mejor ? `${(mejor['F1 Last'] * 100).toFixed(1)}%` : '—', sub: mejor?.Model || '' },
    { label: 'CV walk-forward folds', value: '3', sub: '~24 partidos por test' },
    { label: 'Features pre-partido', value: '25', sub: 'top ANOVA-F · incluye ELO' },
  ];
  $('kpis').innerHTML = cards.map(c => `
    <div class="bg-surface-800 border border-surface-700 rounded-xl p-4">
      <div class="text-xs text-slate-400">${c.label}</div>
      <div class="text-2xl font-bold mt-1">${c.value}</div>
      <div class="text-xs text-slate-500 mt-1">${c.sub}</div>
    </div>
  `).join('');
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------
$('form-predecir').addEventListener('submit', async (e) => {
  e.preventDefault();
  const equipo1 = $('sel-e1').value;
  const equipo2 = $('sel-e2').value;
  const fase    = $('sel-fase').value;
  const n_runs  = parseInt($('inp-runs').value, 10) || 20;

  if (equipo1 === equipo2) { alert('Elige dos equipos distintos'); return; }

  try {
    setLoading(true, `Prediciendo ${equipo1} vs ${equipo2} (${n_runs} corridas × 6 modelos)...`);
    const r = await api.predecir({ equipo1, equipo2, fase, n_runs, evaluacion_id: state.evaluacionId });
    renderPrediccion(r);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

function renderPrediccion(r) {
  state.ultimaPrediccion = r;
  $('resultado').classList.remove('hidden');
  $('r-e1').textContent = r.equipo1;
  $('r-e2').textContent = r.equipo2;
  $('r-elo1').textContent = Math.round(r.elo_e1);
  $('r-elo2').textContent = Math.round(r.elo_e2);
  const sign = r.diff_elo > 0 ? '+' : '';
  $('r-diff').textContent = `Δ ELO ${sign}${Math.round(r.diff_elo)}`;

  // Forma reciente de cada equipo
  if (r.forma_e1) {
    const f1 = r.forma_e1;
    $('r-forma1').textContent = `Forma 5: ${f1.w}W-${f1.d}D-${f1.l}L (GF ${f1.gf} GC ${f1.gc})`;
  }
  if (r.forma_e2) {
    const f2 = r.forma_e2;
    $('r-forma2').textContent = `Forma 5: ${f2.w}W-${f2.d}D-${f2.l}L (GF ${f2.gf} GC ${f2.gc})`;
  }
  if (r.h2h && r.h2h.n > 0) {
    $('r-h2h').textContent = `H2H ${r.h2h.n}: ${r.h2h.w}-${r.h2h.d}-${r.h2h.l} (${r.h2h.gf}-${r.h2h.gc})`;
  } else {
    $('r-h2h').textContent = 'Sin H2H previo';
  }

  const pred = r.consenso.pred;
  const predColor = pred === 'Win' ? 'text-emerald-400' : pred === 'Loss' ? 'text-rose-400' : 'text-amber-400';
  const predEl = $('r-consenso-pred');
  predEl.className = `text-3xl font-bold tracking-tight ${predColor}`;
  predEl.textContent = pred === 'Win' ? `${r.equipo1}` : pred === 'Loss' ? `${r.equipo2}` : 'Empate';

  // Gráfico consenso (donut)
  if (state.chartConsenso) state.chartConsenso.destroy();
  state.chartConsenso = new Chart($('chart-consenso'), {
    type: 'doughnut',
    data: {
      labels: [`Win ${r.equipo1}`, 'Empate', `Win ${r.equipo2}`],
      datasets: [{
        data: [r.consenso.win, r.consenso.draw, r.consenso.loss],
        backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
        borderColor: '#0b0f1a',
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#cbd5e1', font: { size: 11 } } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${(ctx.raw * 100).toFixed(1)}%` } },
      },
      cutout: '60%',
    },
  });

  // Tabla por modelo
  const tbody = $('tbl-modelos');
  tbody.innerHTML = r.modelos.map(m => {
    const c = m.pred === 'Win' ? 'text-emerald-400' : m.pred === 'Loss' ? 'text-rose-400' : 'text-amber-400';
    return `
      <tr class="border-b border-surface-700/50">
        <td class="py-2 px-2 font-medium">${m.modelo}</td>
        <td class="py-2 px-2 text-center font-mono text-xs ${c}">${m.pred}</td>
        <td class="py-2 px-2 text-right font-mono">${fmtPct(m.win)}</td>
        <td class="py-2 px-2 text-right font-mono">${fmtPct(m.draw)}</td>
        <td class="py-2 px-2 text-right font-mono">${fmtPct(m.loss)}</td>
      </tr>`;
  }).join('');

  // Marcadores
  $('goles').innerHTML = r.goles.map(g => `
    <div class="bg-surface-800 border border-surface-700 rounded-lg p-4 text-center">
      <div class="text-xs text-slate-400">${g.modelo}</div>
      <div class="text-2xl font-bold mt-1 font-mono">
        <span>${r.equipo1}</span>
        <span class="text-brand-500 mx-2">${g.g1} – ${g.g2}</span>
        <span>${r.equipo2}</span>
      </div>
      <div class="text-xs text-slate-500 mt-1 font-mono">±${g.std1.toFixed(1)} / ±${g.std2.toFixed(1)}</div>
    </div>
  `).join('');

  // Scroll suave al resultado
  $('resultado').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------
async function cargarFeatureImportance() {
  try {
    const data = await api.featureImportance(state.evaluacionId, 15);
    const max = Math.max(...data.top_features.map(f => f.importance));
    const tooltips = {
      'ELO_E1': 'Rating ELO del local antes del partido',
      'ELO_E2': 'Rating ELO del visitante antes del partido',
      'Diff_ELO': 'Diferencia de ELO entre los dos equipos',
      'Forma_Pts_E1': 'Puntos del local en sus últimos 5 partidos (3W+D)',
      'Forma_Pts_E2': 'Puntos del visitante en sus últimos 5 partidos',
      'Diff_Forma_Pts': 'Diferencia de puntos de forma reciente',
      'Diff_Forma_GD': 'Diferencia de goal-diff de forma reciente',
      'Dias_Descanso_E1': 'Días de descanso del local',
      'Dias_Descanso_E2': 'Días de descanso del visitante',
      'H2H_W_E1': 'Victorias del local en últimos 3 H2H',
      'H2H_D': 'Empates en últimos 3 H2H',
      'H2H_L_E1': 'Derrotas del local en últimos 3 H2H',
    };
    $('feature-importance').innerHTML = data.top_features.map((f, i) => {
      const pct = (f.importance / max) * 100;
      const isAdvanced = ['ELO_', 'Forma_', 'Diff_', 'Dias_', 'H2H_'].some(p => f.feature.startsWith(p));
      const color = isAdvanced ? 'bg-emerald-500' : 'bg-brand-500';
      const tooltip = tooltips[f.feature] || '';
      return `
        <div class="flex items-center gap-3">
          <div class="w-6 text-xs text-slate-500 font-mono">${i + 1}</div>
          <div class="flex-1">
            <div class="flex justify-between text-xs mb-1">
              <span class="${isAdvanced ? 'text-emerald-400 font-medium' : 'text-slate-300'}" title="${tooltip}">${f.feature}</span>
              <span class="text-slate-500 font-mono">${(f.importance * 100).toFixed(2)}%</span>
            </div>
            <div class="h-1.5 bg-surface-700 rounded-full overflow-hidden">
              <div class="${color} h-full" style="width:${pct}%"></div>
            </div>
          </div>
        </div>`;
    }).join('');
  } catch (err) {
    $('feature-importance').innerHTML = `<div class="text-xs text-slate-500">No disponible: ${err.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Track record público (predicciones guardadas)
// ---------------------------------------------------------------------------
async function cargarTrackPublico() {
  const [lista, stats] = await Promise.all([api.trackList(), api.trackStats()]);

  // KPIs
  const kpis = [
    { label: 'Total predicciones', value: stats.total_predicciones, sub: `${stats.resueltas} resueltas / ${stats.pendientes} pendientes` },
    { label: 'Accuracy histórica', value: stats.resueltas > 0 ? `${(stats.accuracy_global * 100).toFixed(1)}%` : '—', sub: stats.resueltas > 0 ? `${stats.aciertos} de ${stats.resueltas}` : 'sin datos aún' },
    { label: 'Precisión por clase', value: '', sub: ['Win', 'Draw', 'Loss'].map(c => `${c}: ${(stats.por_clase_predicha[c].accuracy * 100).toFixed(0)}%`).join(' · ') },
  ];
  $('track-kpis').innerHTML = kpis.map(c => `
    <div class="bg-surface-900/50 border border-surface-700 rounded-lg p-3">
      <div class="text-xs text-slate-400">${c.label}</div>
      <div class="text-xl font-bold mt-1">${c.value}</div>
      <div class="text-xs text-slate-500 mt-0.5">${c.sub}</div>
    </div>
  `).join('');

  // Tabla
  const tbody = $('tbl-track-publico');
  if (lista.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-6 text-slate-500 text-xs">Aún no has guardado predicciones — usa el botón "Guardar para track record" después de predecir.</td></tr>`;
    return;
  }
  tbody.innerHTML = lista.slice().reverse().map(t => {
    let estado = '<span class="text-amber-400">Pendiente</span>';
    let real = '—';
    if (t.resultado_real) {
      real = `${t.resultado_real[0]} <span class="text-slate-500 font-mono text-xs">${t.g1_real}-${t.g2_real}</span>`;
      estado = t.acierto
        ? '<span class="text-emerald-400">✓ Acertó</span>'
        : '<span class="text-rose-400">✗ Falló</span>';
    }
    const probaMax = Math.max(t.prob_win, t.prob_draw, t.prob_loss);
    const predColor = t.pred_consenso === 'Win' ? 'text-emerald-400' : t.pred_consenso === 'Loss' ? 'text-rose-400' : 'text-amber-400';
    return `
      <tr class="border-b border-surface-700/50">
        <td class="py-2 px-2 font-mono text-xs text-slate-400">${t.fecha_partido || t.fecha_prediccion}</td>
        <td class="py-2 px-2">${t.equipo1} <span class="text-slate-500">vs</span> ${t.equipo2}</td>
        <td class="py-2 px-2 text-center font-mono text-xs ${predColor}">${t.pred_consenso}</td>
        <td class="py-2 px-2 text-right font-mono text-xs">${(probaMax * 100).toFixed(0)}%</td>
        <td class="py-2 px-2 text-center text-xs">${real}</td>
        <td class="py-2 px-2 text-center text-xs">${estado}</td>
      </tr>`;
  }).join('');
}

// Botón "Guardar al track record" (en el resultado de predicción)
$('btn-guardar-track').addEventListener('click', async () => {
  if (!state.ultimaPrediccion) return;
  const r = state.ultimaPrediccion;
  const fecha = prompt('Fecha del partido (YYYY-MM-DD), o deja vacío:', '');
  try {
    setLoading(true, 'Guardando al track record...');
    await api.trackCrear({
      equipo1: r.equipo1,
      equipo2: r.equipo2,
      fecha_partido: fecha || null,
      evaluacion_id: state.evaluacionId,
      n_runs: r.n_runs,
      fase: r.fase,
    });
    await cargarTrackPublico();
    alert(`✓ Predicción guardada: ${r.equipo1} vs ${r.equipo2} → ${r.consenso.pred}`);
    document.querySelector('#tbl-track-publico').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ---------------------------------------------------------------------------
// Botón reentrenar
// ---------------------------------------------------------------------------
$('btn-sync').addEventListener('click', async () => {
  try {
    setLoading(true, 'Sincronizando dataset desde Excel...');
    const res = await fetch(`${API_BASE}/api/sync`, { method: 'POST' });
    const data = await res.json();
    await cargarPartidos();
    renderKpis();
    alert(`Sync completo: ${data.nuevos_agregados} partidos nuevos. Total: ${data.total_partidos}`);
  } catch (err) {
    alert(`Error al sincronizar: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

$('btn-evaluar').addEventListener('click', async () => {
  if (!confirm('Reentrenar el modelo desde cero? (1-2 min)')) return;
  try {
    setLoading(true, 'Reentrenando pipeline completo...');
    const nuevaEv = await api.crearEvaluacion();
    state.evaluacionId = nuevaEv.id;
    $('eval-id-text').textContent = `#${state.evaluacionId}`;
    await Promise.all([
      cargarPartidos(), cargarMetricas(), cargarRanking(),
      cargarTrackRecord(), cargarTrackPublico(), cargarFeatureImportance(),
    ]);
    renderKpis();
    $('resultado').classList.add('hidden');
  } catch (err) {
    alert(`Error al reentrenar: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

document.addEventListener('DOMContentLoaded', init);
