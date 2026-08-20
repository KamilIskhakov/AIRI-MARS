#!/usr/bin/env python3
"""Build a standalone HTML viewer for substitution annotation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Substitution Annotation Viewer</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #d9dee7;
      --accent: #155eef;
      --changed: #b42318;
      --preserved: #027a48;
      --uncertain: #9a6700;
      --chip: #eef2f7;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 250, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px 12px;
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1.6fr) repeat(6, minmax(130px, 1fr)) auto auto;
      gap: 8px;
      align-items: center;
    }
    input, select, button {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      color: var(--text);
      padding: 7px 9px;
      font: inherit;
    }
    button {
      cursor: pointer;
      white-space: nowrap;
      background: var(--text);
      color: white;
      border-color: var(--text);
      padding-inline: 12px;
    }
    button.secondary {
      background: white;
      color: var(--text);
      border-color: var(--line);
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 430px;
      gap: 14px;
      padding: 14px 18px 18px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
    }
    .stat .value { font-size: 22px; font-weight: 750; line-height: 1.1; }
    .stat .label { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .charts {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .chart, .table-wrap, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .chart { padding: 10px 12px; min-height: 150px; }
    .chart h2, aside h2 {
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 116px minmax(60px, 1fr) 42px;
      gap: 8px;
      align-items: center;
      margin: 7px 0;
      font-size: 12px;
    }
    .bar-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #374151;
    }
    .bar-track {
      height: 9px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 999px; }
    .table-wrap { overflow: hidden; }
    .table-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid #eef1f5;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #f8fafc;
      color: #4b5563;
      font-size: 12px;
      font-weight: 650;
      position: sticky;
      top: 104px;
      z-index: 5;
    }
    tbody tr { cursor: pointer; }
    tbody tr:hover, tbody tr.active { background: #eef5ff; }
    .truncate {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--chip);
      color: #344054;
      font-size: 12px;
      font-weight: 650;
    }
    .pill.changed { color: var(--changed); background: #fff1f0; }
    .pill.preserved { color: var(--preserved); background: #ecfdf3; }
    .pill.uncertain { color: var(--uncertain); background: #fffaeb; }
    aside {
      position: sticky;
      top: 118px;
      height: calc(100vh - 136px);
      overflow: auto;
      padding: 12px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }
    .meta {
      background: #f8fafc;
      border: 1px solid #eef1f5;
      border-radius: 7px;
      padding: 7px 8px;
      font-size: 12px;
    }
    .meta b { display: block; color: var(--muted); font-weight: 600; margin-bottom: 2px; }
    .context {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: white;
      line-height: 1.45;
      font-size: 13px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin-bottom: 10px;
    }
    mark {
      background: #fff3b0;
      padding: 0 2px;
      border-radius: 3px;
    }
    .rationale {
      padding: 10px;
      background: #f8fafc;
      border: 1px solid #eef1f5;
      border-radius: 8px;
      line-height: 1.45;
      font-size: 13px;
    }
    .empty {
      color: var(--muted);
      padding: 16px;
    }
    @media (max-width: 1180px) {
      .toolbar { grid-template-columns: 1fr 1fr 1fr; }
      main { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .stats { grid-template-columns: repeat(3, 1fr); }
      .charts { grid-template-columns: 1fr; }
      th { position: static; }
    }
    @media (max-width: 720px) {
      header { padding: 12px; }
      main { padding: 12px; }
      .toolbar, .stats { grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: 1fr; }
      table { min-width: 860px; }
      .table-scroll { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Substitution Annotation Viewer</h1>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Поиск: entity, candidate, rationale, context">
      <select id="labelFilter"></select>
      <select id="kindFilter"></select>
      <select id="providerFilter"></select>
      <select id="coarseFilter"></select>
      <select id="fineFilter"></select>
      <select id="domainFilter"></select>
      <button id="hideIdentity" type="button">Hide Identity</button>
      <button id="hideExact" type="button">Hide Exact</button>
      <button id="onlyMismatch" class="secondary" type="button">Mismatch</button>
      <button id="randomPick" type="button">Random</button>
    </div>
  </header>

  <main>
    <section>
      <div class="stats" id="stats"></div>
      <div class="charts">
        <div class="chart"><h2>Candidate Quality</h2><div id="labelChart"></div></div>
        <div class="chart"><h2>Candidate Kinds</h2><div id="kindChart"></div></div>
        <div class="chart"><h2>Providers</h2><div id="providerChart"></div></div>
      </div>
      <div class="table-wrap">
        <div class="table-meta">
          <span id="tableCount"></span>
          <span>Клик по строке открывает контекст справа. Показаны первые <span id="limitLabel"></span> строк фильтра.</span>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th style="width:80px">ID</th>
                <th style="width:120px">Quality</th>
                <th style="width:170px">Kind</th>
                <th style="width:155px">Entity</th>
                <th style="width:155px">Candidate</th>
                <th style="width:110px">Type</th>
                <th style="width:110px">Provider</th>
                <th>Short Context</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <aside>
      <h2>Selected Pair</h2>
      <div id="detail" class="empty">Выбери строку или нажми Random.</div>
    </aside>
  </main>

  <script id="summary-data" type="application/json">__SUMMARY__</script>
  <script id="rows-data" type="application/json">__ROWS__</script>
  <script>
    const allRows = JSON.parse(document.getElementById('rows-data').textContent);
    const summary = JSON.parse(document.getElementById('summary-data').textContent);
    const displayLimit = 500;
    let activeId = null;
    let mismatchOnly = false;
    let hideIdentity = true;
    let hideExact = true;

    const $ = (id) => document.getElementById(id);
    const controls = ['search', 'labelFilter', 'kindFilter', 'providerFilter', 'coarseFilter', 'fineFilter', 'domainFilter'];

    function expectedLabel(row) {
      if (row.expected_score === null || row.expected_score === undefined) return '';
      const value = Number(row.expected_score);
      if (value >= 0.9) return 'preserved';
      if (value <= 0.1) return 'changed';
      return 'uncertain';
    }

    function isMismatch(row) {
      const expected = expectedLabel(row);
      return expected && row.judge_label !== expected;
    }

    function normLiteral(value) {
      return String(value ?? '').normalize('NFKC').toLocaleUpperCase().replace(/\s+/g, ' ').trim();
    }

    function isExactCandidate(row) {
      return normLiteral(row.entity) === normLiteral(row.candidate);
    }

    function qualityLabel(label) {
      if (label === 'preserved') return 'good';
      if (label === 'changed') return 'bad';
      return label || 'unknown';
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      })[ch]);
    }

    function highlight(text, token) {
      const safe = escapeHtml(text ?? '');
      if (!token) return safe;
      const needle = String(token).trim();
      if (!needle) return safe;
      const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return safe.replace(new RegExp(escaped, 'gi'), (m) => `<mark>${m}</mark>`);
    }

    function countBy(rows, key) {
      const out = new Map();
      for (const row of rows) {
        const value = row[key] || 'unknown';
        out.set(value, (out.get(value) || 0) + 1);
      }
      return [...out.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
    }

    function populateSelect(id, label, values) {
      const select = $(id);
      select.innerHTML = `<option value="">${label}: all</option>` +
        values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');
    }

    function initFilters() {
      populateSelect('labelFilter', 'Quality', countBy(allRows, 'judge_label').map(([v]) => v));
      populateSelect('kindFilter', 'Kind', countBy(allRows, 'candidate_kind').map(([v]) => v));
      populateSelect('providerFilter', 'Provider', countBy(allRows, 'provider').map(([v]) => v));
      populateSelect('coarseFilter', 'Coarse', countBy(allRows, 'coarse_group').map(([v]) => v));
      populateSelect('fineFilter', 'Fine', countBy(allRows, 'fine_type').map(([v]) => v));
      populateSelect('domainFilter', 'Domain', countBy(allRows, 'domain').map(([v]) => v));
      for (const id of controls) {
        $(id).addEventListener('input', render);
        $(id).addEventListener('change', render);
      }
      $('onlyMismatch').addEventListener('click', () => {
        mismatchOnly = !mismatchOnly;
        $('onlyMismatch').classList.toggle('secondary', !mismatchOnly);
        render();
      });
      $('hideIdentity').addEventListener('click', () => {
        hideIdentity = !hideIdentity;
        $('hideIdentity').classList.toggle('secondary', !hideIdentity);
        render();
      });
      $('hideExact').addEventListener('click', () => {
        hideExact = !hideExact;
        $('hideExact').classList.toggle('secondary', !hideExact);
        render();
      });
      $('randomPick').addEventListener('click', () => {
        const rows = filteredRows();
        if (!rows.length) return;
        selectRow(rows[Math.floor(Math.random() * rows.length)].pair_id);
      });
      $('limitLabel').textContent = displayLimit;
    }

    function filteredRows() {
      const query = $('search').value.trim().toLowerCase();
      const label = $('labelFilter').value;
      const kind = $('kindFilter').value;
      const provider = $('providerFilter').value;
      const coarse = $('coarseFilter').value;
      const fine = $('fineFilter').value;
      const domain = $('domainFilter').value;
      return allRows.filter((row) => {
        if (label && row.judge_label !== label) return false;
        if (kind && row.candidate_kind !== kind) return false;
        if (provider && row.provider !== provider) return false;
        if (coarse && row.coarse_group !== coarse) return false;
        if (fine && row.fine_type !== fine) return false;
        if (domain && row.domain !== domain) return false;
        if (hideIdentity && row.candidate_kind === 'identity') return false;
        if (hideExact && isExactCandidate(row)) return false;
        if (mismatchOnly && !isMismatch(row)) return false;
        if (!query) return true;
        const haystack = [
          row.pair_id, row.entity, row.candidate, row.short_2w, row.judge_rationale,
          row.original_context, row.candidate_context, row.dataset, row.fine_type, row.candidate_kind
        ].join(' ').toLowerCase();
        return haystack.includes(query);
      });
    }

    function pct(part, total) {
      if (!total) return '0%';
      return `${((part / total) * 100).toFixed(1)}%`;
    }

    function renderStats(rows) {
      const total = rows.length;
      const labels = Object.fromEntries(countBy(rows, 'judge_label'));
      const mismatches = rows.filter(isMismatch).length;
      const trainReady = rows.filter((r) => r.judge_label === 'preserved' || r.judge_label === 'changed').length;
      const avgScoreRows = rows.filter((r) => typeof r.judge_score === 'number');
      const avgScore = avgScoreRows.length
        ? (avgScoreRows.reduce((sum, r) => sum + Number(r.judge_score), 0) / avgScoreRows.length).toFixed(3)
        : 'n/a';
      const items = [
        ['Filtered', total.toLocaleString()],
        ['Preserved', (labels.preserved || 0).toLocaleString()],
        ['Changed', (labels.changed || 0).toLocaleString()],
        ['Uncertain', (labels.uncertain || 0).toLocaleString()],
        ['Train-ready', trainReady.toLocaleString()],
        ['Mismatch', `${mismatches.toLocaleString()} (${pct(mismatches, total)})`],
      ];
      $('stats').innerHTML = items.map(([label, value]) => `
        <div class="stat"><div class="value">${escapeHtml(value)}</div><div class="label">${escapeHtml(label)}</div></div>
      `).join('');
      $('tableCount').textContent = `${total.toLocaleString()} filtered from ${allRows.length.toLocaleString()} total; avg score ${avgScore}`;
    }

    function renderChart(id, rows, key, maxItems = 8) {
      const counts = countBy(rows, key).slice(0, maxItems);
      const max = counts.length ? counts[0][1] : 1;
      $(id).innerHTML = counts.map(([label, count]) => `
        <div class="bar-row" title="${escapeHtml(label)}: ${count}">
          <div class="bar-label">${escapeHtml(label)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (count / max) * 100)}%"></div></div>
          <div>${count}</div>
        </div>
      `).join('') || '<div class="empty">No rows</div>';
    }

    function rowClass(label) {
      return label === 'preserved' ? 'preserved' : label === 'changed' ? 'changed' : 'uncertain';
    }

    function renderTable(rows) {
      const shown = rows.slice(0, displayLimit);
      $('rows').innerHTML = shown.map((row) => `
        <tr data-id="${escapeHtml(row.pair_id)}" class="${row.pair_id === activeId ? 'active' : ''}">
          <td>${escapeHtml(row.pair_id)}</td>
          <td><span class="pill ${rowClass(row.judge_label)}">${escapeHtml(qualityLabel(row.judge_label))}</span></td>
          <td><div class="truncate">${escapeHtml(row.candidate_kind)}</div></td>
          <td><div class="truncate" title="${escapeHtml(row.entity)}">${escapeHtml(row.entity)}</div></td>
          <td><div class="truncate" title="${escapeHtml(row.candidate)}">${escapeHtml(row.candidate)}</div></td>
          <td><div class="truncate">${escapeHtml(row.fine_type)}</div></td>
          <td><div class="truncate">${escapeHtml(row.provider)}</div></td>
          <td><div class="truncate" title="${escapeHtml(row.short_2w)}">${escapeHtml(row.short_2w)}</div></td>
        </tr>
      `).join('');
      for (const tr of $('rows').querySelectorAll('tr')) {
        tr.addEventListener('click', () => selectRow(tr.dataset.id));
      }
    }

    function selectRow(pairId) {
      activeId = pairId;
      const row = allRows.find((item) => item.pair_id === pairId);
      if (!row) return;
      renderDetail(row);
      renderTable(filteredRows());
    }

    function renderDetail(row) {
      const expected = expectedLabel(row) || 'none';
      const mismatch = isMismatch(row) ? 'yes' : 'no';
      $('detail').className = '';
      $('detail').innerHTML = `
        <div class="detail-grid">
          <div class="meta"><b>Pair</b>${escapeHtml(row.pair_id)}</div>
          <div class="meta"><b>Quality / Score</b><span class="pill ${rowClass(row.judge_label)}">${escapeHtml(qualityLabel(row.judge_label))}</span> ${escapeHtml(row.judge_score)}</div>
          <div class="meta"><b>Expected / Mismatch</b>${escapeHtml(expected)} / ${escapeHtml(mismatch)}</div>
          <div class="meta"><b>Kind</b>${escapeHtml(row.candidate_kind)}</div>
          <div class="meta"><b>Entity</b>${escapeHtml(row.entity)}</div>
          <div class="meta"><b>Candidate</b>${escapeHtml(row.candidate)}</div>
          <div class="meta"><b>Type</b>${escapeHtml(row.coarse_group)} / ${escapeHtml(row.fine_type)}</div>
          <div class="meta"><b>Provider</b>${escapeHtml(row.provider)}</div>
          <div class="meta"><b>Dataset</b>${escapeHtml(row.dataset)}</div>
          <div class="meta"><b>Domain</b>${escapeHtml(row.domain)}</div>
        </div>
        <h2>Judge Rationale</h2>
        <div class="rationale">${escapeHtml(row.judge_rationale)}</div>
        <h2 style="margin-top:14px">Original Context</h2>
        <div class="context">${highlight(row.original_context, row.entity)}</div>
        <h2>Candidate Context</h2>
        <div class="context">${highlight(row.candidate_context, row.candidate)}</div>
      `;
    }

    function render() {
      const rows = filteredRows();
      renderStats(rows);
      renderChart('labelChart', rows, 'judge_label');
      renderChart('kindChart', rows, 'candidate_kind');
      renderChart('providerChart', rows, 'provider');
      renderTable(rows);
      if (!activeId && rows.length) selectRow(rows[0].pair_id);
      if (activeId && !rows.some((r) => r.pair_id === activeId)) {
        activeId = rows[0]?.pair_id || null;
        if (activeId) renderDetail(rows[0]);
        else $('detail').innerHTML = '<div class="empty">Нет строк под текущие фильтры.</div>';
      }
    }

    initFilters();
    render();
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__ROWS__", rows_json).replace("__SUMMARY__", summary_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"rows={len(rows)} wrote={args.out} size_mb={args.out.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
