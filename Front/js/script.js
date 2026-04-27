// ── Global State & Config ──
const API_BASE = 'http://localhost:8000/api/v1'; // Adjust if using /api/v1 prefix
const LABELS = ['Upload', 'Preview', 'Configure', 'Results'];
let state = { 
    columns: [], 
    jobId: null, 
    dtypes: {} 
};

// ── Navigation Logic ──
function showStep(stepNum) {
    // Toggle visibility of steps
    document.querySelectorAll('.step').forEach((el, index) => {
        el.classList.toggle('active', (index + 1) === stepNum);
    });
    
    // Update progress dots/label
    document.querySelectorAll('.step-dot').forEach((dot, index) => {
        dot.className = 'step-dot';
        if (index + 1 === stepNum) dot.classList.add('active');
        if (index + 1 < stepNum) dot.classList.add('done');
    });
    
    document.getElementById('progress-label').textContent = `Step ${stepNum}: ${LABELS[stepNum - 1]}`;
}

function resetApp() {
    state = { columns: [], jobId: null, dtypes: {} };
    document.getElementById('upload-form').reset();
    document.getElementById('upload-error').innerText = '';
    document.getElementById('config-error').innerText = '';
    showStep(1);
}

// ── Step 1: Upload Logic ──
async function handleUpload(event) {
    event.preventDefault();
    
    const btn = document.getElementById('upload-btn');
    const errorDiv = document.getElementById('upload-error');
    const formData = new FormData(event.target);

    btn.disabled = true;
    btn.textContent = 'Uploading...';
    errorDiv.innerText = '';

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Save metadata to state
        state.columns = data.columns;
        state.jobId = data.job_id;
        state.dtypes = data.dtypes || {}; // Expecting {col: "float64", ...}
        
        populateTargetDropdown(data.columns, state.dtypes);
        renderTable(data.columns, data.rows);
        
        showStep(2);
    } catch (error) {
        errorDiv.innerText = error.message;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Upload';
    }
}

// ── Step 2: Render Table ──
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

// ── Step 3: Config Logic ──
function toggleTargetField() {
    const isClustering = document.getElementById('task-select').value === 'clustering';
    document.getElementById('target-section').style.display = isClustering ? 'none' : 'block';
}

function populateTargetDropdown(columns, dtypes) {
    const select = document.getElementById('target-select');
    select.innerHTML = columns.map(c => {
        const typeLabel = dtypes[c] ? ` (${dtypes[c]})` : '';
        return `<option value="${c}">${c}${typeLabel}</option>`;
    }).join('');
    
    // Default to the last column (common for datasets)
    select.selectedIndex = columns.length - 1;
}

async function startTraining() {
    const btn = document.querySelector('#step-3 button');
    const errorDiv = document.getElementById('config-error');
    
    const payload = {
        job_id: state.jobId,
        task_type: document.getElementById('task-select').value,
        target_column: document.getElementById('target-select').value
    };

    btn.disabled = true;
    btn.textContent = "Training Models...";
    errorDiv.innerText = '';

    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Training failed on server.");
        }

        const report = await response.json(); 
        renderResults(report);
        showStep(4);
    } catch (err) {
        errorDiv.innerText = err.message;
    } finally {
        btn.disabled = false;
        btn.textContent = "Train Models";
    }
}

// ── Step 4: Results Comparison Logic ──
function renderResults(report) {
    const container = document.getElementById('results-content');
    if (!report || !report.all_results) return;

    // 1. Header with the "Winner"
    let html = `
        <div class="winner-announcement" style="background: #e8f5e9; padding: 15px; border-radius: 8px; border: 1px solid #2e7d32; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1b5e20;"> Best Model: ${report.best_model}</h2>
            <p style="margin: 5px 0 0 0;">${report.message}</p>
        </div>
        <div class="comparison-grid" style="display: flex; gap: 20px; flex-wrap: wrap;">
    `;

    // 2. Map through all results for side-by-side comparison
    report.all_results.forEach(res => {
        html += `
            <div class="model-card" style="flex: 1; min-width: 300px; background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
                <h3 style="margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px;">${res.name}</h3>
                <div class="metrics">
        `;

        // Classification Metrics
        if (res.accuracy !== undefined) {
            html += generateMetricRow("Accuracy", (res.accuracy * 100).toFixed(2) + "%");
            if (res.cm) {
                html += `<details style="margin-top: 10px; cursor: pointer;">
                            <summary>View Confusion Matrix</summary>
                            <pre style="font-size: 10px; background: #eee; padding: 5px;">${JSON.stringify(res.cm, null, 2)}</pre>
                         </details>`;
            }
        } 
        // Regression Metrics
        else if (res.mse !== undefined) {
            html += generateMetricRow("MSE", res.mse.toFixed(4));
            html += generateMetricRow("MAE", res.mae ? res.mae.toFixed(4) : "N/A");
            html += generateMetricRow("R² Score", res.r2.toFixed(4));
        }
        // Clustering Metrics
        else if (res.silhouette !== undefined) {
            const score = res.silhouette === -1 ? "N/A (Invalid Clusters)" : res.silhouette.toFixed(4);
            html += generateMetricRow("Silhouette Score", score);
        }

        html += `</div></div>`;
    });

    html += `</div>`;
    container.innerHTML = html;
}

// Helper to keep the HTML clean
function generateMetricRow(label, value) {
    return `
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
            <span style="color: #666;">${label}</span>
            <strong style="color: #2c3e50;">${value}</strong>
        </div>`;
}

async function saveModel() {
    try {
        const response = await fetch(`${API_BASE}/download/${state.jobId}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Download failed .");
        }

        const file = await response.blob();
        const url = window.URL.createObjectURL(file);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `model_pipeline_${state.jobId}.pkl`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error(error);
    }
}
