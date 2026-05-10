from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import json
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .model_registry import ModelRegistry
from .schemas import PredictRequest, RegisterModelRequest

app = FastAPI(
    title='Sentiment Model Registry API',
    version='1.0.0',
    description='Локальный API для управления моделями, версиями и предсказаниями тональности.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

registry = ModelRegistry()


@app.get('/health')
def health():
    return {'status': 'ok', 'models': len(registry.list_models())}


@app.get('/api/v1/models')
def list_models():
    return {'models': registry.list_models()}


@app.get('/api/v1/families')
def list_families():
    return {'families': registry.list_families()}


@app.get('/api/v1/models/{model_id}')
def get_model(model_id: str):
    try:
        return registry.get_model(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/api/v1/models/register')
def register_model(req: RegisterModelRequest):
    try:
        model = registry.register_model(
            family=req.family,
            source=req.source,
            source_type=req.source_type,
            description=req.description,
            active=req.active,
            version=req.version,
        )
        return {'model': model}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/v1/models/{family}/activate/{version}')
def activate_version(family: str, version: int):
    try:
        model = registry.set_active(family, version)
        return {'model': model}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete('/api/v1/models/{family}/versions/{version}')
def delete_version(family: str, version: int):
    try:
        registry.delete_version(family, version)
        return {'status': 'deleted', 'family': family, 'version': version}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/api/v1/models/{model_id}/export')
def export_model(model_id: str, destination: str):
    try:
        path = registry.export_model(model_id, destination)
        return {'destination': path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/v1/predict')
def predict(req: PredictRequest):
    try:
        result = registry.predict(
            text=req.text,
            model_id=req.model_id,
            family=req.family,
            version=req.version,
            top_k=req.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/v1/models/{model_id}/predict')
def predict_by_model_id(model_id: str, req: PredictRequest):
    try:
        result = registry.predict(
            text=req.text,
            model_id=model_id,
            top_k=req.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/v1/predict/batch')
async def predict_batch(
    file: UploadFile = File(...),
    text_column: str = 'text',
    family: Optional[str] = None,
    version: Optional[int] = None,
    model_id: Optional[str] = None,
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found")

        rows = []
        for idx, text in enumerate(df[text_column].fillna('').astype(str).tolist()):
            pred = registry.predict(text=text, family=family, version=version, model_id=model_id, top_k=1)
            rows.append({
                'row': idx,
                'text': text,
                'label': pred['prediction']['label'],
                'score': pred['prediction']['score'],
                'model_family': pred['model']['family'],
                'model_version': pred['model']['version'],
            })
        return {'results': rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get('/', response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sentiment Service Pro</title>
  <style>
    :root {
      /* Современная палитра */
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --accent: #4f46e5; /* Indigo */
      --accent-hover: #4338ca;
      --secondary: #64748b;
      --success: #10b981;
      --code-bg: #1e293b;
      --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      line-height: 1.5;
    }

    .page {
      max-width: 800px; /* Ограничиваем ширину для лучшей читаемости в один ряд */
      margin: 0 auto;
      padding: 40px 20px;
    }

    header {
      text-align: center;
      margin-bottom: 40px;
    }

    h1 {
      margin: 0 0 12px;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.025em;
    }

    .subtitle {
      margin: 0;
      color: var(--text-muted);
      font-size: 16px;
    }

    .topbar {
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-top: 20px;
    }

    .topbar a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      font-size: 14px;
      transition: color 0.2s;
    }

    .topbar a:hover {
      color: var(--accent-hover);
      text-decoration: underline;
    }

    /* Настройки высоты панелей */
    .settings-bar {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      box-shadow: var(--shadow);
    }

    .settings-bar label {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-muted);
    }

    input[type="range"] {
      flex-grow: 1;
      accent-color: var(--accent);
    }

    /* Вертикальный стэк карточек */
    .stack {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 28px;
      box-shadow: var(--shadow);
    }

    .card h2 {
      margin: 0 0 20px;
      font-size: 20px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .card h2::before {
      content: '';
      width: 4px;
      height: 20px;
      background: var(--accent);
      border-radius: 4px;
    }

    .field {
      margin-bottom: 20px;
    }

    .field label {
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 600;
      color: var(--text-main);
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      font: inherit;
      font-size: 14px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }

    textarea {
      resize: vertical;
      min-height: 100px;
    }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 12px 24px;
      font-size: 15px;
      font-weight: 700;
      transition: background 0.2s, transform 0.1s;
    }

    button:hover {
      background: var(--accent-hover);
    }

    button:active {
      transform: scale(0.98);
    }

    button.secondary {
      background: #f1f5f9;
      color: #475569;
      border: 1px solid var(--border);
      width: auto;
    }

    button.secondary:hover {
      background: #e2e8f0;
    }

    /* Вывод JSON */
    .output-container {
      margin-top: 20px;
    }

    .output-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    .output {
      background: var(--code-bg);
      color: #e2e8f0;
      border-radius: 12px;
      padding: 16px;
      font-family: 'Fira Code', 'Cascadia Code', monospace;
      font-size: 13px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-all;
      line-height: 1.6;
      border: 1px solid #334155;
    }

    /* Таблицы версий */
    .family-block {
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-top: 16px;
      overflow: hidden;
    }

    .family-head {
      padding: 12px 16px;
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      font-weight: 700;
      font-size: 14px;
    }

    .family-head .active-tag {
      background: #dcfce7;
      color: #166534;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 12px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
    }

    th, td {
      padding: 12px 16px;
      text-align: left;
      font-size: 13px;
      border-bottom: 1px solid var(--border);
    }

    th {
      background: #ffffff;
      color: var(--text-muted);
      font-weight: 600;
    }

    .actions {
      display: flex;
      gap: 6px;
    }

    .actions button {
      padding: 6px 12px;
      font-size: 12px;
      width: auto;
    }

    .actions button.btn-delete {
      background: #fee2e2;
      color: #991b1b;
    }

    .actions button.btn-delete:hover {
      background: #fecaca;
    }

    .hint {
      font-size: 13px;
      color: var(--text-muted);
      margin: 10px 0;
      font-style: italic;
    }

    @media (max-width: 600px) {
      .settings-bar { flex-direction: column; align-items: stretch; }
    }
  </style>
</head>
<body>

  <div class="page">
    <header>
      <h1>Sentiment Service</h1>
      <p class="subtitle">Интерфейс управления моделями и анализа тональности текста</p>
      <div class="topbar">
        <a href="/docs" target="_blank">Swagger API</a>
        <a href="/health" target="_blank">System Health</a>
      </div>
    </header>

    <!-- Контроль высоты -->
    <div class="settings-bar">
      <label for="panelHeight">Высота панелей:</label>
      <input id="panelHeight" type="range" min="120" max="600" step="20" value="240" oninput="setPanelHeight(this.value)">
      <span id="panelHeightValue">240 px</span>
    </div>

    <div class="stack">
      
      <!-- Регистрация -->
      <section class="card">
        <h2>Register model version</h2>
        <div class="field">
          <label>Family</label>
          <input id="family" value="base-rubert">
        </div>
        <div class="field">
          <label>Source type</label>
          <select id="source_type">
            <option value="hf">Hugging Face (hf)</option>
            <option value="local">Local Path</option>
          </select>
        </div>
        <div class="field">
          <label>Source / Repo ID</label>
          <input id="source" value="cointegrated/rubert-tiny2">
        </div>
        <div class="field">
          <label>Description</label>
          <input id="description" placeholder="Краткое описание версии (опционально)">
        </div>
        <button onclick="registerModel()">Зарегистрировать версию</button>

        <div class="output-container">
          <div class="output-title">API Response</div>
          <pre id="registerOut" class="output">{}</pre>
        </div>
      </section>

      <!-- Предикт -->
      <section class="card">
        <h2>Predict review</h2>
        <div class="field">
          <label>Model family</label>
          <input id="pred_family" value="base-rubert">
        </div>
        <div class="field">
          <label>Version</label>
          <input id="pred_version" placeholder="Оставьте пустым для активной версии">
        </div>
        <div class="field">
          <label>Текст для анализа</label>
          <textarea id="text" rows="4">Товар пришёл быстро, качество отличное, я довольна покупкой.</textarea>
        </div>
        <button onclick="predict()">Выполнить анализ</button>

        <div class="output-container">
          <div class="output-title">Prediction Result</div>
          <pre id="predictOut" class="output">{}</pre>
        </div>
      </section>

      <!-- Список моделей -->
      <section class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
          <h2 style="margin: 0;">Families and versions</h2>
          <button class="secondary" onclick="refreshFamilies()">Обновить данные</button>
        </div>
        <p class="hint">Управляйте доступными версиями. Активная версия используется для запросов по умолчанию.</p>
        <div id="families" class="families"></div>
      </section>

      <!-- Batch -->
      <section class="card">
        <h2>Batch CSV Processing</h2>
        <div class="field">
          <label>Загрузить CSV файл</label>
          <input type="file" id="csvFile">
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
          <div class="field">
            <label>Колонка с текстом</label>
            <input id="csvTextCol" value="text">
          </div>
          <div class="field">
            <label>Model Family</label>
            <input id="batch_family" value="base-rubert">
          </div>
        </div>
        <button onclick="batchPredict()">Запустить пакетную обработку</button>

        <div class="output-container">
          <div class="output-title">Batch Results</div>
          <pre id="batchOut" class="output">{}</pre>
        </div>
      </section>

    </div>
  </div>

<script>
/* JS ЛОГИКА ОСТАЛАСЬ БЕЗ ИЗМЕНЕНИЙ (ФУНКЦИОНАЛЬНОСТЬ СОХРАНЕНА) */

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function safePrettyText(text) {
  try {
    return pretty(JSON.parse(text));
  } catch (e) {
    return text;
  }
}

function setPanelHeight(value) {
  const px = `${value}px`;
  document.getElementById('panelHeightValue').textContent = px;
  document.querySelectorAll('.output').forEach(el => {
    el.style.maxHeight = px;
  });
}

async function registerModel() {
  const payload = {
    family: document.getElementById('family').value,
    source_type: document.getElementById('source_type').value,
    source: document.getElementById('source').value,
    description: document.getElementById('description').value || null,
    active: true
  };

  const res = await fetch('/api/v1/models/register', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });

  const text = await res.text();
  document.getElementById('registerOut').textContent = safePrettyText(text);
  refreshFamilies();
}

async function predict() {
  const versionRaw = document.getElementById('pred_version').value.trim();

  const payload = {
    text: document.getElementById('text').value,
    family: document.getElementById('pred_family').value || null,
    version: versionRaw ? parseInt(versionRaw, 10) : null,
    top_k: 3
  };

  const res = await fetch('/api/v1/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });

  const text = await res.text();
  document.getElementById('predictOut').textContent = safePrettyText(text);
}

async function refreshFamilies() {
  const res = await fetch('/api/v1/families');
  const data = await res.json();
  const el = document.getElementById('families');

  if (!data.families || data.families.length === 0) {
    el.innerHTML = '<p class="hint">No models registered yet.</p>';
    return;
  }

  let html = '';
  for (const fam of data.families) {
    html += `<div class="family-block">`;
    html += `<div class="family-head">
               <span>📁 ${fam.family}</span>
               <span class="active-tag">active v${fam.active_version ?? 'none'}</span>
             </div>`;
    html += '<table><thead><tr><th>Ver</th><th>Status</th><th>Source</th><th>Actions</th></tr></thead><tbody>';

    for (const v of fam.versions) {
      html += `<tr>
        <td><strong>${v.version}</strong></td>
        <td>${v.active ? '✅' : '—'}</td>
        <td title="${v.source}"><code style="font-size:11px">${v.source}</code></td>
        <td>
          <div class="actions">
            <button onclick="activateVersion('${fam.family}', ${v.version})">Activate</button>
            <button class="btn-delete" onclick="deleteVersion('${fam.family}', ${v.version})">Del</button>
          </div>
        </td>
      </tr>`;
    }

    html += '</tbody></table></div>';
  }

  el.innerHTML = html;
}

async function activateVersion(family, version) {
  const res = await fetch(`/api/v1/models/${encodeURIComponent(family)}/activate/${version}`, {
    method: 'POST'
  });
  const text = await res.text();
  alert(text);
  refreshFamilies();
}

async function deleteVersion(family, version) {
  if(!confirm('Удалить эту версию?')) return;
  const res = await fetch(`/api/v1/models/${encodeURIComponent(family)}/versions/${version}`, {
    method: 'DELETE'
  });
  const text = await res.text();
  alert(text);
  refreshFamilies();
}

async function batchPredict() {
  const fileInput = document.getElementById('csvFile');
  if (!fileInput.files.length) {
    alert('Choose a CSV file');
    return;
  }

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('text_column', document.getElementById('csvTextCol').value || 'text');
  form.append('family', document.getElementById('batch_family').value || '');

  const res = await fetch('/api/v1/predict/batch', {
    method: 'POST',
    body: form
  });

  const text = await res.text();
  document.getElementById('batchOut').textContent = safePrettyText(text);
}

setPanelHeight(document.getElementById('panelHeight').value);
refreshFamilies();
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get('/api/v1/models/{family}/versions')
def family_versions(family: str):
    try:
        fam = next(x for x in registry.list_families() if x['family'] == family)
        return fam
    except StopIteration:
        raise HTTPException(status_code=404, detail=f'Family not found: {family}')
