/**
 * SENTINEL — Application Logic & Security Interface Controller
 * Multi-Layer Phishing Detection & Risk Assessment
 */

(function () {
  'use strict';

  // State Management
  const state = {
    currentRoute: 'overview',
    backendOnline: false,
    modelMetrics: null,
    scanHistory: [],
    activeFilter: 'all',
    searchQuery: '',
    currentScanResult: null
  };

  // Local Storage Key
  const STORAGE_KEY = 'sentinel_scan_history_v2';

  // SVG Icons
  const ICONS = {
    check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`,
    alert: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`,
    copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`
  };

  // -------------------------------------------------------------------------
  // 1. Initialization
  // -------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    loadScanHistory();
    setupNavigation();
    setupScannerEvents();
    setupHistoryControls();
    setupLightbox();
    checkBackendHealth();
    loadModelMetrics();

    // Default route handling
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
  });

  // -------------------------------------------------------------------------
  // 2. Navigation & Routing
  // -------------------------------------------------------------------------
  function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        const route = link.getAttribute('data-route');
        if (route) {
          e.preventDefault();
          window.location.hash = route;
        }
      });
    });

    // Mobile drawer toggle
    const toggleBtn = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('drawer-open');
      });

      // Close drawer on click outside
      document.addEventListener('click', (e) => {
        if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target) && sidebar.classList.contains('drawer-open')) {
          sidebar.classList.remove('drawer-open');
        }
      });
    }

    // Header Quick Scan button
    const headerScanBtn = document.getElementById('header-scan-btn');
    if (headerScanBtn) {
      headerScanBtn.addEventListener('click', () => {
        window.location.hash = 'scanner';
        setTimeout(() => {
          const input = document.getElementById('scanner-url-input');
          if (input) input.focus();
        }, 100);
      });
    }
  }

  function handleHashChange() {
    const hash = (window.location.hash || '#overview').replace('#', '').trim();
    const validRoutes = ['overview', 'scanner', 'analytics', 'history', 'about'];
    const targetRoute = validRoutes.includes(hash) ? hash : 'overview';

    state.currentRoute = targetRoute;

    // Update active nav links
    document.querySelectorAll('.nav-link').forEach(link => {
      if (link.getAttribute('data-route') === targetRoute) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Update active page view
    document.querySelectorAll('.page-view').forEach(view => {
      if (view.id === `view-${targetRoute}`) {
        view.classList.add('active');
      } else {
        view.classList.remove('active');
      }
    });

    // Update breadcrumb
    const breadcrumb = document.getElementById('active-breadcrumb');
    if (breadcrumb) {
      const titles = {
        overview: 'System Overview',
        scanner: 'Live URL Scanner',
        analytics: 'Model Analytics & Benchmarks',
        history: 'Audit Scan History',
        about: 'Project Architecture & Documentation'
      };
      breadcrumb.textContent = titles[targetRoute] || 'Dashboard';
    }

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Close mobile drawer if open
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('drawer-open');

    // If history route, re-render history table
    if (targetRoute === 'history') {
      renderHistoryTable();
    }
  }

  // -------------------------------------------------------------------------
  // 3. Backend Health & Metrics Integration
  // -------------------------------------------------------------------------
  async function checkBackendHealth() {
    const statusDot = document.getElementById('engine-pulse-dot');
    const statusText = document.getElementById('engine-status-text');
    const systemBadge = document.getElementById('system-badge');

    try {
      const resp = await fetch('/api/health');
      if (resp.ok) {
        const data = await resp.json();
        state.backendOnline = data.model_ready;

        if (statusDot) {
          statusDot.style.background = '#10b981';
          statusDot.style.boxShadow = '0 0 8px #10b981';
        }
        if (statusText) {
          statusText.textContent = `Online: ${data.model_name || 'Gradient Boosting'}`;
        }
        if (systemBadge) {
          systemBadge.innerHTML = `<span>●</span> 28 Features Active`;
          systemBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          systemBadge.style.color = '#6ee7b7';
        }
      } else {
        throw new Error('Degraded API response');
      }
    } catch (err) {
      state.backendOnline = false;
      if (statusDot) {
        statusDot.style.background = '#f59e0b';
        statusDot.style.boxShadow = '0 0 8px #f59e0b';
      }
      if (statusText) {
        statusText.textContent = 'API Connecting (Local)';
      }
    }
  }

  async function loadModelMetrics() {
    try {
      const resp = await fetch('/api/model-metrics');
      if (resp.ok) {
        const data = await resp.json();
        state.modelMetrics = data;
        populateMetricsData(data);
      }
    } catch (err) {
      console.warn('Could not load dynamic model metrics:', err);
    }
  }

  function populateMetricsData(data) {
    if (!data || !data.report) return;
    const best = data.report.best_model_metrics;
    if (!best) return;

    // Overview cards
    const accEl = document.getElementById('metric-accuracy-val');
    const precEl = document.getElementById('metric-precision-val');
    const recEl = document.getElementById('metric-recall-val');
    const rocAucEl = document.getElementById('metric-roc-val');

    if (accEl) accEl.textContent = `${(best.accuracy * 100).toFixed(2)}%`;
    if (precEl) precEl.textContent = `${(best.precision * 100).toFixed(2)}%`;
    if (recEl) recEl.textContent = `${(best.recall * 100).toFixed(2)}%`;
    if (rocAucEl) rocAucEl.textContent = best.roc_auc.toFixed(4);

    // Populate comparison table in analytics view
    renderAnalyticsTable(data.comparison_table || data.report.all_models_ranked);
  }

  function renderAnalyticsTable(models) {
    const tbody = document.getElementById('model-comparison-tbody');
    if (!tbody || !models || !models.length) return;

    tbody.innerHTML = '';
    models.forEach((m, idx) => {
      const isBest = idx === 0 || m.model_name === 'Gradient Boosting';
      const tr = document.createElement('tr');
      if (isBest) tr.classList.add('highlight-best');

      const acc = (parseFloat(m.accuracy) * 100).toFixed(2);
      const prec = (parseFloat(m.precision) * 100).toFixed(2);
      const rec = (parseFloat(m.recall) * 100).toFixed(2);
      const f1 = (parseFloat(m.f1_score) * 100).toFixed(2);
      const rocAuc = parseFloat(m.roc_auc).toFixed(4);
      const fpr = (parseFloat(m.false_positive_rate || 0) * 100).toFixed(2);
      const timeSec = parseFloat(m.training_time_seconds || 0).toFixed(2);

      tr.innerHTML = `
        <td>
          <strong>${escapeHtml(m.model_name)}</strong>
          ${isBest ? '<span class="best-model-pill">SELECTED BEST</span>' : ''}
        </td>
        <td><strong>${acc}%</strong></td>
        <td>${prec}%</td>
        <td>${rec}%</td>
        <td>${f1}%</td>
        <td>${rocAuc}</td>
        <td>${fpr}%</td>
        <td>${timeSec}s</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------------------
  // 4. URL Scanner Engine & Events
  // -------------------------------------------------------------------------
  function setupScannerEvents() {
    // Both Overview scanner and dedicated Scanner elements
    const inputs = [
      document.getElementById('overview-url-input'),
      document.getElementById('scanner-url-input')
    ].filter(Boolean);

    inputs.forEach(input => {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          executeScan(input.value.trim(), input.id.includes('overview') ? 'overview' : 'scanner');
        }
      });
    });

    // Analyze buttons
    const overviewBtn = document.getElementById('overview-analyze-btn');
    if (overviewBtn) {
      overviewBtn.addEventListener('click', () => {
        const val = document.getElementById('overview-url-input')?.value.trim();
        executeScan(val, 'overview');
      });
    }

    const scannerBtn = document.getElementById('scanner-analyze-btn');
    if (scannerBtn) {
      scannerBtn.addEventListener('click', () => {
        const val = document.getElementById('scanner-url-input')?.value.trim();
        executeScan(val, 'scanner');
      });
    }

    // Clear buttons
    document.querySelectorAll('.btn-clear-input').forEach(btn => {
      btn.addEventListener('click', () => {
        inputs.forEach(i => i.value = '');
        inputs[0]?.focus();
      });
    });

    // Paste buttons
    document.querySelectorAll('.btn-paste-input').forEach(btn => {
      btn.addEventListener('click', async () => {
        try {
          if (navigator.clipboard && navigator.clipboard.readText) {
            const text = await navigator.clipboard.readText();
            if (text) {
              inputs.forEach(i => i.value = text.trim());
              showToast('URL pasted from clipboard', 'info');
            }
          } else {
            showToast('Clipboard access not permitted in browser', 'error');
          }
        } catch (err) {
          showToast('Could not read from clipboard', 'error');
        }
      });
    });

    // Sample URL shortcut buttons
    document.querySelectorAll('.sample-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        const sampleUrl = pill.getAttribute('data-url');
        if (sampleUrl) {
          inputs.forEach(i => i.value = sampleUrl);
          // If on overview, scan immediately or navigate to scanner
          const activeContext = state.currentRoute === 'scanner' ? 'scanner' : 'overview';
          executeScan(sampleUrl, activeContext);
        }
      });
    });

    // Feature Drawer Toggle in results
    document.querySelectorAll('.feature-drawer-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetDrawer = btn.nextElementSibling;
        if (targetDrawer) {
          targetDrawer.classList.toggle('open');
          const chevron = btn.querySelector('.chevron-icon');
          if (chevron) {
            chevron.style.transform = targetDrawer.classList.contains('open') ? 'rotate(180deg)' : 'rotate(0)';
          }
        }
      });
    });
  }

  async function executeScan(rawUrl, sourceContext) {
    if (!rawUrl || !rawUrl.trim()) {
      showToast('Please enter a valid URL to analyze.', 'error');
      return;
    }

    const url = rawUrl.trim();

    // UI Loading state
    setScanningState(true, sourceContext);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with status ${response.status}`);
      }

      const result = await response.json();
      state.currentScanResult = result;

      // Persist in Scan History
      saveScanRecord(result);

      // Render Result Card in the active source
      renderScanResult(result, sourceContext);
      showToast(`Scan complete: ${result.classification} (${result.risk_level} RISK)`, result.risk_level === 'HIGH' ? 'error' : 'success');

    } catch (err) {
      console.error('Scan execution error:', err);
      showToast(err.message || 'Analysis failed. Check local server connection.', 'error');
    } finally {
      setScanningState(false, sourceContext);
    }
  }

  function setScanningState(isLoading, sourceContext) {
    const loadingState = document.getElementById(`${sourceContext}-loading`);
    const resultCard = document.getElementById(`${sourceContext}-result`);
    const analyzeBtn = document.getElementById(`${sourceContext}-analyze-btn`);

    if (loadingState) {
      loadingState.classList.toggle('active', isLoading);
    }
    if (resultCard && isLoading) {
      resultCard.classList.remove('active');
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = isLoading;
      analyzeBtn.innerHTML = isLoading
        ? `<div class="radar-spinner" style="width:18px;height:18px;margin:0;border-width:2px;"></div> Analyzing...`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> Analyze URL`;
    }
  }

  function renderScanResult(data, sourceContext) {
    const card = document.getElementById(`${sourceContext}-result`);
    if (!card) return;

    // Reset theme classes
    card.classList.remove('result-high', 'result-medium', 'result-low');
    const riskLower = (data.risk_level || 'low').toLowerCase();
    card.classList.add(`result-${riskLower}`);

    // Badges & Status
    const classBadge = card.querySelector('.classification-badge');
    if (classBadge) {
      classBadge.className = `classification-badge badge-${riskLower}`;
      classBadge.innerHTML = `${data.classification === 'PHISHING' ? ICONS.alert : ICONS.check} ${escapeHtml(data.classification)}`;
    }

    const riskTag = card.querySelector('.risk-level-tag');
    if (riskTag) {
      riskTag.textContent = `${data.risk_level} RISK LEVEL`;
    }

    const urlEl = card.querySelector('.result-url-text');
    if (urlEl) {
      urlEl.textContent = data.url;
    }

    // Gauge Score & Probability
    const scoreValEl = card.querySelector('.gauge-numeric');
    if (scoreValEl) {
      scoreValEl.className = `gauge-numeric score-${riskLower}`;
      scoreValEl.textContent = Math.round(data.risk_score || 0);
    }

    const probValEl = card.querySelector('.gauge-prob-val');
    if (probValEl) {
      probValEl.textContent = `${(data.phishing_probability * 100).toFixed(1)}%`;
    }

    // Reason Summary
    const explanationEl = card.querySelector('.result-explanation-text');
    if (explanationEl) {
      explanationEl.textContent = data.reason_summary || 'Standard risk assessment completed.';
    }

    // Suspicious Indicators List
    const warningsList = card.querySelector('.warnings-signal-list');
    if (warningsList) {
      warningsList.innerHTML = '';
      const warnings = data.warnings || [];
      if (warnings.length === 0) {
        warningsList.innerHTML = `<li class="signal-item" style="color:var(--text-muted);">No structural anomalies detected</li>`;
      } else {
        warnings.forEach(w => {
          const li = document.createElement('li');
          li.className = 'signal-item warn-item';
          li.innerHTML = `${ICONS.alert} <span>${escapeHtml(w)}</span>`;
          warningsList.appendChild(li);
        });
      }
    }

    // Reassuring Signals List
    const safeList = card.querySelector('.safe-signal-list');
    if (safeList) {
      safeList.innerHTML = '';
      const safeSignals = data.safe_signals || [];
      if (safeSignals.length === 0) {
        safeList.innerHTML = `<li class="signal-item" style="color:var(--text-muted);">No reassuring structural traits identified</li>`;
      } else {
        safeSignals.forEach(s => {
          const li = document.createElement('li');
          li.className = 'signal-item safe-item';
          li.innerHTML = `${ICONS.check} <span>${escapeHtml(s)}</span>`;
          safeList.appendChild(li);
        });
      }
    }

    // Extracted 28 Features Drawer
    const drawer = card.querySelector('.features-grid-drawer');
    if (drawer && data.features) {
      drawer.innerHTML = '';
      Object.entries(data.features).forEach(([featName, featVal]) => {
        const cell = document.createElement('div');
        cell.className = 'feature-pill-cell';
        cell.innerHTML = `
          <span class="feature-pill-name">${escapeHtml(featName)}</span>
          <span class="feature-pill-val">${escapeHtml(String(featVal))}</span>
        `;
        drawer.appendChild(cell);
      });
    }

    // Scan Time & Timestamp
    const scanTimeEl = card.querySelector('.scan-time-badge');
    if (scanTimeEl) {
      scanTimeEl.textContent = `${data.scan_time_ms} ms (offline)`;
    }

    // Copy URL Action
    const copyBtn = card.querySelector('.btn-copy-url');
    if (copyBtn) {
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(data.url).then(() => {
          showToast('URL copied to clipboard', 'info');
        });
      };
    }

    // Reveal card
    card.classList.add('active');
  }

  // -------------------------------------------------------------------------
  // 5. Scan History Management (localStorage)
  // -------------------------------------------------------------------------
  function loadScanHistory() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      state.scanHistory = stored ? JSON.parse(stored) : [];
    } catch (e) {
      console.warn('Could not read scan history from localStorage:', e);
      state.scanHistory = [];
    }
    renderOverviewRecentScans();
  }

  function saveScanRecord(result) {
    if (!result || !result.url) return;

    const record = {
      id: Date.now(),
      url: result.url,
      classification: result.classification,
      risk_level: result.risk_level,
      risk_score: result.risk_score,
      phishing_probability: result.phishing_probability,
      warnings_count: (result.warnings || []).length,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      date: new Date().toLocaleDateString(),
      full_result: result
    };

    // Prepend to list & limit to 50 most recent records
    state.scanHistory.unshift(record);
    if (state.scanHistory.length > 50) {
      state.scanHistory = state.scanHistory.slice(0, 50);
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.scanHistory));
    } catch (e) {
      console.warn('Storage limit exceeded, trimming history:', e);
    }

    renderOverviewRecentScans();
    if (state.currentRoute === 'history') {
      renderHistoryTable();
    }
  }

  function renderOverviewRecentScans() {
    const listEl = document.getElementById('overview-recent-list');
    const emptyEl = document.getElementById('overview-recent-empty');
    if (!listEl) return;

    const recent = state.scanHistory.slice(0, 4);

    if (recent.length === 0) {
      if (emptyEl) emptyEl.style.display = 'block';
      listEl.innerHTML = '';
      return;
    }

    if (emptyEl) emptyEl.style.display = 'none';
    listEl.innerHTML = '';

    recent.forEach(r => {
      const riskClass = (r.risk_level || 'low').toLowerCase();
      const div = document.createElement('div');
      div.className = 'glass-panel';
      div.style.padding = '14px 18px';
      div.style.marginBottom = '10px';
      div.style.cursor = 'pointer';

      div.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span class="classification-badge badge-${riskClass}" style="font-size:0.75rem;padding:3px 10px;">
            ${escapeHtml(r.classification)} • ${escapeHtml(r.risk_level)}
          </span>
          <span style="font-size:0.75rem;color:var(--text-muted);">${escapeHtml(r.timestamp)}</span>
        </div>
        <div style="font-family:var(--font-mono);font-size:0.85rem;color:var(--text-highlight);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
          ${escapeHtml(r.url)}
        </div>
      `;

      div.addEventListener('click', () => {
        window.location.hash = 'scanner';
        setTimeout(() => {
          const input = document.getElementById('scanner-url-input');
          if (input) input.value = r.url;
          renderScanResult(r.full_result, 'scanner');
        }, 100);
      });

      listEl.appendChild(div);
    });
  }

  function setupHistoryControls() {
    const searchInput = document.getElementById('history-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        state.searchQuery = e.target.value.toLowerCase().trim();
        renderHistoryTable();
      });
    }

    const filterBtns = document.querySelectorAll('.history-filter-btn');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeFilter = btn.getAttribute('data-filter') || 'all';
        renderHistoryTable();
      });
    });

    const clearBtn = document.getElementById('btn-clear-history');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        if (state.scanHistory.length === 0) {
          showToast('Scan history is already empty', 'info');
          return;
        }
        if (confirm('Are you sure you want to clear your local scan history?')) {
          state.scanHistory = [];
          localStorage.removeItem(STORAGE_KEY);
          renderHistoryTable();
          renderOverviewRecentScans();
          showToast('Scan history cleared', 'info');
        }
      });
    }

    const exportBtn = document.getElementById('btn-export-history');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        if (state.scanHistory.length === 0) {
          showToast('No scan history to export', 'error');
          return;
        }
        const blob = new Blob([JSON.stringify(state.scanHistory, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `sentinel_scan_history_${Date.now()}.json`;
        a.click();
        showToast('Scan history exported as JSON', 'success');
      });
    }
  }

  function renderHistoryTable() {
    const tbody = document.getElementById('history-table-tbody');
    const emptyState = document.getElementById('history-empty-state');
    const tableWrap = document.getElementById('history-table-wrapper');
    if (!tbody) return;

    let filtered = state.scanHistory.filter(item => {
      const matchesSearch = !state.searchQuery || item.url.toLowerCase().includes(state.searchQuery);
      let matchesFilter = true;
      if (state.activeFilter === 'high') matchesFilter = item.risk_level === 'HIGH';
      else if (state.activeFilter === 'medium') matchesFilter = item.risk_level === 'MEDIUM';
      else if (state.activeFilter === 'low') matchesFilter = item.risk_level === 'LOW';
      else if (state.activeFilter === 'phishing') matchesFilter = item.classification === 'PHISHING';
      else if (state.activeFilter === 'legitimate') matchesFilter = item.classification === 'LEGITIMATE';

      return matchesSearch && matchesFilter;
    });

    if (filtered.length === 0) {
      if (emptyState) emptyState.style.display = 'block';
      if (tableWrap) tableWrap.style.display = 'none';
      tbody.innerHTML = '';
      return;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (tableWrap) tableWrap.style.display = 'block';
    tbody.innerHTML = '';

    filtered.forEach(r => {
      const riskClass = (r.risk_level || 'low').toLowerCase();
      const tr = document.createElement('tr');

      tr.innerHTML = `
        <td style="font-size:0.8rem;color:var(--text-muted);">${escapeHtml(r.timestamp)}</td>
        <td style="font-family:var(--font-mono);max-width:340px;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(r.url)}">
          ${escapeHtml(r.url)}
        </td>
        <td>
          <span class="classification-badge badge-${riskClass}" style="font-size:0.75rem;padding:3px 10px;">
            ${escapeHtml(r.classification)}
          </span>
        </td>
        <td>
          <span style="font-weight:700;" class="score-${riskClass}">${escapeHtml(r.risk_level)}</span>
        </td>
        <td style="font-family:var(--font-mono);font-weight:700;">${Math.round(r.risk_score || 0)} / 100</td>
        <td>
          <button class="btn-input-utility btn-inspect-row" style="padding:4px 10px;font-size:0.75rem;">
            Inspect
          </button>
        </td>
      `;

      tr.querySelector('.btn-inspect-row').addEventListener('click', () => {
        window.location.hash = 'scanner';
        setTimeout(() => {
          const input = document.getElementById('scanner-url-input');
          if (input) input.value = r.url;
          renderScanResult(r.full_result, 'scanner');
        }, 100);
      });

      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------------------
  // 6. Visualizations Gallery Lightbox
  // -------------------------------------------------------------------------
  function setupLightbox() {
    const modal = document.getElementById('lightbox-modal');
    const modalImg = document.getElementById('lightbox-img');
    const modalTitle = document.getElementById('lightbox-title');
    const closeBtn = document.getElementById('lightbox-close-btn');

    if (!modal) return;

    document.querySelectorAll('.gallery-card').forEach(card => {
      card.addEventListener('click', () => {
        const img = card.querySelector('img');
        const title = card.querySelector('.gallery-title')?.textContent || 'Chart Inspection';
        if (img && modalImg) {
          modalImg.src = img.src;
          if (modalTitle) modalTitle.textContent = title;
          modal.classList.add('open');
        }
      });
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', () => modal.classList.remove('open'));
    }

    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.classList.contains('open')) {
        modal.classList.remove('open');
      }
    });
  }

  // -------------------------------------------------------------------------
  // 7. Toast Notification Utility
  // -------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `sentinel-toast ${type === 'error' ? 'toast-error' : type === 'success' ? 'toast-success' : ''}`;
    toast.innerHTML = `
      ${type === 'error' ? ICONS.alert : ICONS.check}
      <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 200ms ease';
      setTimeout(() => toast.remove(), 200);
    }, 3500);
  }

  // -------------------------------------------------------------------------
  // 8. Safe HTML Escaping (Prevent Untrusted XSS)
  // -------------------------------------------------------------------------
  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str || '');
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

})();
