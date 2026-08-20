#!/usr/bin/env python3
"""Build a standalone HTML viewer for paired cross-encoder predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--type-statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    type_statistics = json.loads(args.type_statistics.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    payload = json.dumps(
        {"summary": summary, "typeStatistics": type_statistics, "rows": rows},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = TEMPLATE.replace("__PAYLOAD__", payload)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output} rows={len(rows)}")


TEMPLATE = r'''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Сравнение cross-encoder моделей</title>
  <style>
    :root { --text:#202124; --muted:#5f6368; --line:#dadce0; --soft:#f8f9fa; --blue:#1a73e8; --green:#137333; --green-bg:#e6f4ea; --red:#b3261e; --red-bg:#fce8e6; --amber:#8a4b08; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--text); background:#fff; font:14px/1.45 Arial, sans-serif; }
    header { border-bottom:1px solid var(--line); padding:24px 28px 18px; }
    h1 { margin:0 0 6px; font-size:24px; font-weight:600; }
    h2 { margin:0 0 12px; font-size:17px; }
    p { margin:0; }
    .muted { color:var(--muted); }
    main { padding:20px 28px 36px; max-width:1600px; margin:auto; }
    .metrics { display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); border:1px solid var(--line); border-radius:6px; margin-bottom:20px; }
    .metric { padding:14px 16px; border-right:1px solid var(--line); }
    .metric:last-child { border-right:0; }
    .metric b { display:block; font-size:21px; margin-top:3px; }
    .metric .up { color:var(--green); }
    .coverage { margin:0 0 22px; }
    .coverage-note { margin:-5px 0 12px; color:var(--muted); }
    .coverage table, .subtypes table { border:1px solid var(--line); }
    .bar-cell { min-width:125px; font-variant-numeric:tabular-nums; }
    .bar { height:5px; margin-top:5px; background:#eef0f2; overflow:hidden; }
    .bar span { display:block; height:100%; background:var(--blue); }
    .bar.train span { background:var(--green); }
    .status-missing { color:var(--red); }
    .status-ready { color:var(--green); }
    .subtypes { margin:22px 0; }
    .branch-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:22px 0; }
    .branch-grid section { min-width:0; }
    .branch-grid .table-wrap { max-height:430px; border:1px solid var(--line); }
    .toolbar { display:grid; grid-template-columns:minmax(240px,2fr) repeat(4,minmax(130px,1fr)) auto; gap:8px; margin:14px 0; }
    input, select, button { min-height:36px; border:1px solid var(--line); border-radius:4px; background:#fff; color:var(--text); padding:7px 10px; font:inherit; }
    button { cursor:pointer; }
    button:hover { border-color:#9aa0a6; background:var(--soft); }
    .layout { display:grid; grid-template-columns:minmax(650px,1.45fr) minmax(380px,.85fr); gap:18px; align-items:start; }
    .panel { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .panel-head { display:flex; align-items:center; justify-content:space-between; padding:11px 14px; border-bottom:1px solid var(--line); background:var(--soft); }
    .table-wrap { overflow:auto; max-height:690px; }
    table { width:100%; border-collapse:collapse; }
    th { position:sticky; top:0; z-index:1; background:#fff; text-align:left; color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); padding:9px 10px; white-space:nowrap; }
    td { padding:9px 10px; border-bottom:1px solid #eee; vertical-align:top; }
    tbody tr { cursor:pointer; }
    tbody tr:hover, tbody tr.active { background:#eef4fd; }
    .entity { font-weight:600; max-width:210px; }
    .candidate { max-width:210px; }
    .score { font-variant-numeric:tabular-nums; white-space:nowrap; }
    .pill { display:inline-block; border-radius:999px; padding:2px 8px; white-space:nowrap; font-size:12px; }
    .fixed { color:var(--green); background:var(--green-bg); }
    .regressed { color:var(--red); background:var(--red-bg); }
    .wrong { color:var(--amber); background:#fef7e0; }
    .correct { color:#3c4043; background:#f1f3f4; }
    .detail { padding:15px; max-height:745px; overflow:auto; }
    .detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
    .detail-grid div { border-bottom:1px solid var(--line); padding-bottom:8px; }
    .detail-grid span { display:block; color:var(--muted); font-size:12px; }
    .detail-grid b { display:block; margin-top:2px; word-break:break-word; }
    .context { border-top:1px solid var(--line); padding-top:12px; margin-top:12px; }
    .context h3 { font-size:13px; margin:0 0 7px; }
    .context-text { white-space:pre-wrap; max-height:235px; overflow:auto; background:var(--soft); border:1px solid var(--line); border-radius:4px; padding:10px; font-family:Arial,sans-serif; }
    mark { background:#fdd663; padding:0 2px; }
    .positive { color:var(--green); }
    .negative { color:var(--red); }
    @media (max-width:1050px) { .metrics { grid-template-columns:repeat(3,1fr); } .metric:nth-child(3) { border-right:0; } .layout, .branch-grid { grid-template-columns:1fr; } .detail { max-height:none; } }
    @media (max-width:700px) { header, main { padding-left:14px; padding-right:14px; } .metrics { grid-template-columns:repeat(2,1fr); } .metric:nth-child(odd) { border-right:1px solid var(--line); } .metric:nth-child(even) { border-right:0; } .toolbar { grid-template-columns:1fr 1fr; } .toolbar input { grid-column:1/-1; } .layout { display:block; } .panel + .panel { margin-top:14px; } }
  </style>
</head>
<body>
  <header>
    <h1>Сравнение моделей оценки семантической замены</h1>
    <p class="muted">Одинаковые примеры, сущности не встречались в обучении обеих моделей</p>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="coverage">
      <h2>Покрытие типов данных по этапам</h2>
      <p class="coverage-note" id="coverage-note"></p>
      <div class="table-wrap"><table><thead><tr><th>Группа</th><th>Размеченный инвентарь</th><th>Пилотная разметка</th><th>Надёжный train</th><th>Общая проверка моделей</th><th>Статус метрики</th></tr></thead><tbody id="coverage"></tbody></table></div>
    </section>
    <section class="subtypes">
      <h2>Собственные имена: подтипы и качество</h2>
      <p class="coverage-note">В инвентаре указаны уникальные сущности, на следующих этапах — пары «контекст + кандидат».</p>
      <div class="table-wrap"><table><thead><tr><th>Подтип</th><th>Инвентарь</th><th>Пилотные пары</th><th>Train</th><th>Проверка</th><th>Старая F1</th><th>Новая F1</th><th>Изменение</th></tr></thead><tbody id="proper-types"></tbody></table></div>
    </section>
    <div class="branch-grid">
      <section>
        <h2>Числовые сущности</h2>
        <p class="coverage-note">Размечались в пилоте через короткое контекстное окно, но не вошли в обучение этой модели.</p>
        <div class="table-wrap"><table><thead><tr><th>Подтип</th><th>Инвентарь</th><th>Пилотные пары</th><th>Train / test</th></tr></thead><tbody id="numeric-types"></tbody></table></div>
      </section>
      <section>
        <h2>Обычные и доменные сущности</h2>
        <p class="coverage-note">Для них есть только небольшой пилот; метрик cross-encoder пока нет.</p>
        <div class="table-wrap"><table><thead><tr><th>Подтип</th><th>Инвентарь</th><th>Пилотные пары</th><th>Train / test</th></tr></thead><tbody id="common-types"></tbody></table></div>
      </section>
    </div>
    <section>
      <h2>Предсказания</h2>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Поиск сущности, кандидата или текста">
        <select id="outcome"></select>
        <select id="type"></select>
        <select id="label"><option value="">Оба класса</option><option value="1">Подходит</option><option value="0">Не подходит</option></select>
        <select id="kind"><option value="">Все виды кандидатов</option></select>
        <button id="random">Случайный пример</button>
      </div>
      <div class="layout">
        <div class="panel">
          <div class="panel-head"><span id="count"></span><span class="muted">score «подходит»</span></div>
          <div class="table-wrap"><table><thead><tr><th>Исходная сущность</th><th>Кандидат</th><th>Метка</th><th>Старая</th><th>Новая</th><th>Результат</th><th>Тип</th></tr></thead><tbody id="rows"></tbody></table></div>
        </div>
        <aside class="panel"><div class="panel-head"><b>Разбор примера</b></div><div class="detail" id="detail"></div></aside>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const s = data.summary, ts = data.typeStatistics, all = data.rows;
    const fmt = v => Number(v).toFixed(3);
    const pct = v => `${(100 * Number(v)).toFixed(1)}%`;
    const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const names = {new_fixed:'Новая исправила',new_regressed:'Новая ошиблась',both_correct:'Обе верны',both_wrong:'Обе ошиблись'};
    const classes = {new_fixed:'fixed',new_regressed:'regressed',both_correct:'correct',both_wrong:'wrong'};
    document.getElementById('metrics').innerHTML = [
      ['Общая выборка', s.rows], ['Старая macro-F1', fmt(s.old.macro_f1)], ['Новая macro-F1', fmt(s.new.macro_f1)],
      ['Изменение', `<span class="up">+${(100*s.macro_f1_delta).toFixed(1)} п.п.</span>`],
      ['Исправлено / ухудшено', `${s.outcomes.new_fixed} / ${s.outcomes.new_regressed}`],
      ['95% интервал', `${(100*s.bootstrap.ci95[0]).toFixed(1)}…${(100*s.bootstrap.ci95[1]).toFixed(1)} п.п.`]
    ].map(([a,b]) => `<div class="metric"><span class="muted">${a}</span><b>${b}</b></div>`).join('');

    const stages = ts.stages;
    const groupNames = {proper_name:'Собственные имена',numeric:'Числа',common_entity:'Обычные сущности',domain_term:'Доменные термины',ambiguous:'Неоднозначные',junk:'Шум'};
    const groups = ['proper_name','numeric','common_entity','domain_term','ambiguous','junk'];
    const stageKeys = ['inventory','pilot_annotation','reliable_train','fair_comparison'];
    const maxima = Object.fromEntries(stageKeys.map(k=>[k,Math.max(...groups.map(g=>stages[k].coarse[g]||0),1)]));
    const barCell = (value,max,kind='') => `<td class="bar-cell">${Number(value).toLocaleString('ru-RU')}<div class="bar ${kind}"><span style="width:${100*value/max}%"></span></div></td>`;
    document.getElementById('coverage-note').textContent = `Типизировано ${stages.inventory.total.toLocaleString('ru-RU')} из ${ts.inventory_target.toLocaleString('ru-RU')} сущностей (${pct(ts.inventory_tagged_ratio)}). Инвентарь измеряется сущностями, остальные этапы — парами.`;
    document.getElementById('coverage').innerHTML = groups.map(g=>{
      const inv=stages.inventory.coarse[g]||0, pilot=stages.pilot_annotation.coarse[g]||0, train=stages.reliable_train.coarse[g]||0, test=stages.fair_comparison.coarse[g]||0;
      const status=test?'<span class="status-ready">есть сравнительная метрика</span>':train?'<span class="status-missing">нет общей проверки</span>':'<span class="status-missing">модель не обучалась</span>';
      return `<tr><td><b>${groupNames[g]}</b></td>${barCell(inv,maxima.inventory)}${barCell(pilot,maxima.pilot_annotation)}${barCell(train,maxima.reliable_train,'train')}${barCell(test,maxima.fair_comparison,'train')}<td>${status}</td></tr>`;
    }).join('');

    const nested = (stage,groups) => groups.reduce((acc,g)=>{Object.entries(stages[stage].fine_by_coarse[g]||{}).forEach(([k,v])=>acc[k]=(acc[k]||0)+v); return acc;},{});
    const subtypeTable = (target,groups,withMetrics=false) => {
      const inv=nested('inventory',groups), pilot=nested('pilot_annotation',groups), train=nested('reliable_train',groups), test=nested('fair_comparison',groups);
      const names=[...new Set([...Object.keys(inv),...Object.keys(pilot),...Object.keys(train),...Object.keys(test)])].sort((a,b)=>(inv[b]||0)-(inv[a]||0));
      const maxInv=Math.max(...Object.values(inv),1), maxPilot=Math.max(...Object.values(pilot),1);
      document.getElementById(target).innerHTML=names.map(name=>{
        if (!withMetrics) return `<tr><td><b>${esc(name)}</b></td>${barCell(inv[name]||0,maxInv)}${barCell(pilot[name]||0,maxPilot)}<td class="status-missing">0 / 0</td></tr>`;
        const perf=ts.performance_by_type[name], old=perf?fmt(perf.old_macro_f1):'—', current=perf?fmt(perf.new_macro_f1):'—', delta=perf?perf.new_macro_f1-perf.old_macro_f1:null;
        return `<tr><td><b>${esc(name)}</b></td>${barCell(inv[name]||0,maxInv)}<td>${(pilot[name]||0).toLocaleString('ru-RU')}</td><td>${(train[name]||0).toLocaleString('ru-RU')}</td><td>${(test[name]||0).toLocaleString('ru-RU')}</td><td>${old}</td><td>${current}</td><td class="${delta===null?'muted':delta>=0?'positive':'negative'}">${delta===null?'нет данных':`${delta>=0?'+':''}${(100*delta).toFixed(1)} п.п.`}</td></tr>`;
      }).join('');
    };
    subtypeTable('proper-types',['proper_name'],true);
    subtypeTable('numeric-types',['numeric']);
    subtypeTable('common-types',['common_entity','domain_term']);

    const outcome = document.getElementById('outcome'), type = document.getElementById('type'), kind = document.getElementById('kind');
    outcome.innerHTML = '<option value="">Все результаты</option>' + Object.entries(names).map(([v,n])=>`<option value="${v}">${n}</option>`).join('');
    type.innerHTML = '<option value="">Все типы</option>' + [...new Set(all.map(x=>x.entity_type))].sort().map(v=>`<option>${esc(v)}</option>`).join('');
    kind.innerHTML += [...new Set(all.map(x=>x.pair_kind))].sort().map(v=>`<option>${esc(v)}</option>`).join('');
    let filtered = all, active = null;

    function highlight(text, needle) {
      const safe = esc(text), n = esc(needle); if (!n) return safe;
      const i = safe.toLocaleLowerCase().indexOf(n.toLocaleLowerCase());
      return i < 0 ? safe : safe.slice(0,i)+'<mark>'+safe.slice(i,i+n.length)+'</mark>'+safe.slice(i+n.length);
    }
    function show(row) {
      active = row;
      document.querySelectorAll('tbody tr').forEach(tr=>tr.classList.toggle('active',tr.dataset.id===row._id));
      document.getElementById('detail').innerHTML = `
        <div class="detail-grid">
          <div><span>Исходная сущность</span><b>${esc(row.entity)}</b></div><div><span>Кандидат</span><b>${esc(row.candidate)}</b></div>
          <div><span>Правильная метка</span><b>${row.label ? 'Подходит' : 'Не подходит'}</b></div><div><span>Итог</span><b class="${classes[row.comparison_outcome]}">${names[row.comparison_outcome]}</b></div>
          <div><span>Старая модель</span><b>${fmt(row.old_score)} → ${row.old_prediction ? 'подходит' : 'не подходит'}</b></div><div><span>Новая модель</span><b>${fmt(row.new_score)} → ${row.new_prediction ? 'подходит' : 'не подходит'}</b></div>
          <div><span>Тип</span><b>${esc(row.entity_type)}</b></div><div><span>Вид кандидата</span><b>${esc(row.pair_kind)}</b></div>
        </div>
        <div class="context"><h3>Исходный текст</h3><div class="context-text">${highlight(row.left,row.entity)}</div></div>
        <div class="context"><h3>Текст после подстановки</h3><div class="context-text">${highlight(row.right,row.candidate)}</div></div>`;
    }
    function apply() {
      const q = document.getElementById('search').value.trim().toLocaleLowerCase(), lab = document.getElementById('label').value;
      filtered = all.filter(x => (!q || [x.entity,x.candidate,x.left,x.right].some(v=>String(v??'').toLocaleLowerCase().includes(q))) && (!outcome.value || x.comparison_outcome===outcome.value) && (!type.value || x.entity_type===type.value) && (!kind.value || x.pair_kind===kind.value) && (lab==='' || String(x.label)===lab));
      document.getElementById('count').textContent = `Показано ${Math.min(filtered.length,300)} из ${filtered.length}`;
      document.getElementById('rows').innerHTML = filtered.slice(0,300).map((x,i)=>{x._id=String(all.indexOf(x)); return `<tr data-id="${x._id}" data-pos="${i}"><td class="entity">${esc(x.entity)}</td><td class="candidate">${esc(x.candidate)}</td><td>${x.label?'подходит':'не подходит'}</td><td class="score">${fmt(x.old_score)}</td><td class="score">${fmt(x.new_score)}</td><td><span class="pill ${classes[x.comparison_outcome]}">${names[x.comparison_outcome]}</span></td><td>${esc(x.entity_type)}</td></tr>`}).join('');
      document.querySelectorAll('#rows tr').forEach(tr=>tr.onclick=()=>show(filtered[Number(tr.dataset.pos)]));
      if (filtered.length) show(filtered.includes(active) ? active : filtered[0]); else document.getElementById('detail').innerHTML='<p class="muted">Нет примеров по выбранным фильтрам.</p>';
    }
    ['search','outcome','type','label','kind'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',apply));
    document.getElementById('random').onclick=()=>filtered.length&&show(filtered[Math.floor(Math.random()*filtered.length)]);
    apply();
  </script>
</body>
</html>'''


if __name__ == "__main__":
    main()
