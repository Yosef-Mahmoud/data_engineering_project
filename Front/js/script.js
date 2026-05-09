// ── Global State & Config ──────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000/api/v1';
const LABELS   = ['Upload', 'Preview', 'Configure', 'Results'];

let state = { columns: [], jobId: null, dtypes: {} };


// ── Navigation ─────────────────────────────────────────────────────────────

function showStep(stepNum) {
    document.querySelectorAll('.step').forEach((el, i) => {
        el.classList.toggle('active', i + 1 === stepNum);
    });
    document.querySelectorAll('.step-dot').forEach((dot, i) => {
        dot.className = 'step-dot';
        if (i + 1 === stepNum) dot.classList.add('active');
        if (i + 1 <  stepNum) dot.classList.add('done');
    });
    document.getElementById('progress-label').textContent =
        `Step ${stepNum}: ${LABELS[stepNum - 1]}`;
}

function resetApp() {
    state = { columns: [], jobId: null, dtypes: {} };
    document.getElementById('upload-form').reset();
    document.getElementById('upload-error').innerText = '';
    document.getElementById('config-error').innerText = '';
    document.getElementById('results-content').innerHTML = '';
    showStep(1);
}


// ── Step 1 – Upload ────────────────────────────────────────────────────────

async function handleUpload(event) {
    event.preventDefault();

    const btn      = document.getElementById('upload-btn');
    const errorDiv = document.getElementById('upload-error');
    const formData = new FormData(event.target);

    btn.disabled    = true;
    btn.textContent = 'Uploading…';
    errorDiv.innerText = '';

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server error ${response.status}`);
        }

        const data = await response.json();

        state.columns = data.columns;
        state.jobId   = data.job_id;
        state.dtypes  = data.dtypes || {};

        populateTargetDropdown(data.columns, state.dtypes);
        renderTable(data.columns, data.rows);
        showStep(2);

    } catch (error) {
        errorDiv.innerText = error.message;
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Upload';
    }
}


// ── Step 2 – Preview table ─────────────────────────────────────────────────

function renderTable(columns, rows) {
    const thead = `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
    const tbody = rows.map(row =>
        `<tr>${row.map(cell => `<td>${cell !== null ? cell : ''}</td>`).join('')}</tr>`
    ).join('');

    document.getElementById('preview-table').innerHTML = `
        <table>
            <thead>${thead}</thead>
            <tbody>${tbody}</tbody>
        </table>`;
}


// ── Step 3 – Configuration ─────────────────────────────────────────────────

function toggleTargetField() {
    const isClustering = document.getElementById('task-select').value === 'clustering';
    document.getElementById('target-section').style.display = isClustering ? 'none' : 'block';
}

function populateTargetDropdown(columns, dtypes) {
    const select   = document.getElementById('target-select');
    select.innerHTML = columns.map(c => {
        const typeLabel = dtypes[c] ? ` (${dtypes[c]})` : '';
        return `<option value="${c}">${c}${typeLabel}</option>`;
    }).join('');
    select.selectedIndex = columns.length - 1; // Default to last column (common convention)
}

async function startTraining() {
    const trainBtn = document.getElementById('train-btn');
    const errorDiv = document.getElementById('config-error');
    const taskType = document.getElementById('task-select').value;

    const payload = {
        job_id:    state.jobId,
        task_type: taskType,
        // Only include target_column for supervised tasks; clustering doesn't need it.
        ...(taskType !== 'clustering' && {
            target_column: document.getElementById('target-select').value
        }),
    };

    trainBtn.disabled    = true;
    trainBtn.textContent = 'Training…';
    errorDiv.innerText   = '';

    // Show a spinner in the results panel while we wait.
    document.getElementById('results-content').innerHTML = `
        <div style="text-align:center; padding: 2rem 0;">
            <div class="spinner"></div>
            <p style="color:#6b7280; font-size:0.85rem; margin-top:0.5rem;">
                Training models, please wait…
            </p>
        </div>`;
    showStep(4);

    try {
        const response = await fetch(`${API_BASE}/train`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Training failed on the server.');
        }

        const report = await response.json();
        renderResults(report);

    } catch (err) {
        // Bounce back to step 3 and show the error there.
        showStep(3);
        errorDiv.innerText = err.message;
    } finally {
        trainBtn.disabled    = false;
        trainBtn.textContent = 'Train model';
    }
}


// ── Step 4 – Results ───────────────────────────────────────────────────────

function renderResults(report) {
    const container = document.getElementById('results-content');
    if (!report?.all_results) {
        container.innerHTML = '<p style="color:red">No results returned from server.</p>';
        return;
    }

    let html = `
        <div class="winner-announcement">
            <span class="best-model-tag">🏆 Best model: ${report.best_model}</span>
            <p style="font-size:0.83rem; color:#6b7280; margin:0;">${report.message}</p>
        </div>
        <div class="comparison-grid">`;

    report.all_results.forEach(res => {
        const isWinner  = res.name === report.best_model;
        const winnerCss = isWinner
            ? 'border: 2px solid #34c76f; background: #f0fdf4;'
            : '';

        html += `<div class="model-card" style="${winnerCss}">`;

        if (isWinner) {
            html += `<div style="font-size:0.72rem; font-weight:700; color:#166534;
                        background:#dcfce7; padding:3px 10px; border-radius:6px;
                        display:inline-block; margin-bottom:10px; letter-spacing:.04em;">
                        ✓ BEST
                    </div>`;
        }

        html += `<h3 style="margin:0 0 12px; border-bottom:2px solid #e5e7eb;
                             padding-bottom:10px; font-size:1rem;">${res.name}</h3>
                 <div class="metrics">`;

        // ── Classification ──────────────────────────────────────────────
        if (res.accuracy !== undefined) {
            html += metricRow('Accuracy',  (res.accuracy  * 100).toFixed(2) + '%');
            html += metricRow('Precision', (res.precision * 100).toFixed(2) + '%');
            html += metricRow('Recall',    (res.recall    * 100).toFixed(2) + '%');
            html += metricRow('F1 Score',  (res.f1        * 100).toFixed(2) + '%');

            if (res.cm && res.labels) {
                html += renderConfusionMatrix(res.cm, res.labels);
            }
        }

        // ── Regression ─────────────────────────────────────────────────
        else if (res.mse !== undefined) {
            html += metricRow('R² Score', res.r2.toFixed(4));
            html += metricRow('MAE',      res.mae.toFixed(4));
            html += metricRow('MSE',      res.mse.toFixed(4));
        }

        // ── Clustering ─────────────────────────────────────────────────
        else if (res.silhouette !== undefined) {
            const score = res.silhouette === -1
                ? 'N/A (Invalid clusters)'
                : res.silhouette.toFixed(4);
            html += metricRow('Silhouette Score', score);
        }

        html += `</div></div>`;
    });

    html += `</div>`;
    container.innerHTML = html;
}

// ── Metric row helper ──────────────────────────────────────────────────────

function metricRow(label, value) {
    return `<div style="display:flex; justify-content:space-between;
                         padding:8px 0; border-bottom:1px solid #f0f0f0;">
                <span style="color:#666;">${label}</span>
                <strong style="color:#2c3e50;">${value}</strong>
            </div>`;
}

// ── Confusion matrix as a labelled HTML table ──────────────────────────────
// The original rendered raw JSON in a <pre> tag.  This renders a proper grid
// with Actual/Predicted axis labels — the format shown in Lab 6.

function renderConfusionMatrix(cm, labels) {
    const headerCells = labels.map(l =>
        `<th style="background:#eef1fe; color:#4f6ef7; padding:7px 10px;
                    font-size:0.75rem; font-weight:600;">${l}</th>`
    ).join('');

    const bodyRows = cm.map((row, i) => {
        const cells = row.map((val, j) => {
            const isDiag = i === j;
            return `<td style="padding:7px 10px; text-align:center; font-size:0.82rem;
                        font-weight:${isDiag ? '700' : '400'};
                        background:${isDiag ? '#f0fdf4' : 'transparent'};
                        color:${isDiag ? '#166534' : '#374151'};">${val}</td>`;
        }).join('');
        return `<tr>
                    <th style="background:#f9fafb; padding:7px 10px;
                        font-size:0.75rem; color:#374151; font-weight:600;
                        text-align:left;">${labels[i]}</th>
                    ${cells}
                </tr>`;
    }).join('');

    return `
        <details style="margin-top:12px; cursor:pointer;">
            <summary style="font-size:0.82rem; color:#4f6ef7; font-weight:600;
                            padding:4px 0; list-style:none;">
                ▶ Confusion Matrix
            </summary>
            <div style="overflow-x:auto; margin-top:8px;">
                <table class="matrix-table">
                    <thead>
                        <tr>
                            <th style="background:#f9fafb; font-size:0.72rem;
                                color:#9ca3af; padding:7px 10px;">
                                Act. \\ Pred.
                            </th>
                            ${headerCells}
                        </tr>
                    </thead>
                    <tbody>${bodyRows}</tbody>
                </table>
            </div>
        </details>`;
}


// ── Download model ─────────────────────────────────────────────────────────

async function saveModel() {
    const btn       = document.getElementById('download-btn');
    const errorDiv  = document.getElementById('download-error');
    errorDiv.innerText = '';

    btn.disabled    = true;
    btn.textContent = 'Downloading…';

    try {
        const response = await fetch(`${API_BASE}/download/${state.jobId}`, {
            method: 'GET',
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Download failed.');
        }

        const blob = await response.blob();
        const url  = window.URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.style.display = 'none';
        a.href     = url;
        a.download = `model_pipeline_${state.jobId}.pkl`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

    } catch (error) {
        // Surface the error to the user instead of silently logging.
        errorDiv.innerText = `Download failed: ${error.message}`;
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Download Pipeline';
    }
}