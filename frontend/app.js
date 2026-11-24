/* Cooperative revenue analysis frontend — vanilla JS, no build step. */
"use strict";

const API = "";
const $ = (sel) => document.querySelector(sel);
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#6250d6", "#e34948"];

const state = {
  game: { name: "", players: [], entries: [] },
  analysis: null,
  examples: [],
  activeTaskId: null,
  guarantees: [],   // {value: number|null, locked: boolean} per player
};

// -------------------------------------------------------------------- helpers
// Call convention: api(path) issues GET; api(path, payload) POSTs the payload
// as JSON. Every caller that needs a body passes it as the second argument.
async function api(path, payload) {
  const init = { headers: { "Content-Type": "application/json" } };
  if (payload !== undefined) {
    init.method = "POST";
    init.body = JSON.stringify(payload);
  }
  const res = await fetch(API + path, init);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    const msg = data.error || (data.errors || []).join("；") || "请求失败";
    throw new Error(msg);
  }
  return data;
}

function fmt(x, digits = 4) {
  if (x === null || x === undefined) return "无界";
  if (typeof x !== "number") return String(x);
  if (!isFinite(x)) return "无界";
  if (Math.abs(x) >= 1000) return x.toFixed(1);
  return Number(x.toFixed(digits)).toString();
}

function showMessages(el, items, kind = "info") {
  el.innerHTML = "";
  (Array.isArray(items) ? items : [items]).forEach((text) => {
    const div = document.createElement("div");
    div.className = `msg ${kind}`;
    div.textContent = text;
    el.appendChild(div);
  });
}

function membersToMask(members, players) {
  let mask = 0;
  members.forEach((m) => { mask |= 1 << players.indexOf(m); });
  return mask;
}
function maskToMembers(mask, players) {
  return players.filter((_, i) => mask & (1 << i));
}

// ---------------------------------------------------------------- examples init
async function initExamples() {
  const data = await api("/api/examples");
  state.examples = data.examples;
  const sel = $("#exampleSelect");
  sel.innerHTML = "";
  data.examples.forEach((ex) => {
    const opt = document.createElement("option");
    opt.value = ex.id;
    opt.textContent = ex.title;
    sel.appendChild(opt);
  });
}

$("#loadExampleBtn").addEventListener("click", async () => {
  const id = $("#exampleSelect").value;
  const ex = await api(`/api/examples/${id}`);
  loadGameIntoEditor(ex.game);
  showMessages($("#inputMessages"), `已载入示例：${ex.title}`, "ok");
});

function loadGameIntoEditor(gameDict) {
  state.game = JSON.parse(JSON.stringify(gameDict));
  $("#gameName").value = state.game.name || "";
  $("#playersInput").value = state.game.players.join(", ");
  renderCoalitionTable();
  renderGuaranteeTable();
}

$("#applyPlayersBtn").addEventListener("click", () => {
  const names = $("#playersInput").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean);
  if (names.length === 0) return showMessages($("#inputMessages"), "至少需要一个参与方。", "error");
  if (new Set(names).size !== names.length)
    return showMessages($("#inputMessages"), "参与方名称必须唯一。", "error");
  state.game.players = names;
  state.game.name = $("#gameName").value || state.game.name;
  // drop entries referencing removed players; keep compatible ones
  const keep = [];
  state.game.entries.forEach((e) => {
    if (e.members.every((m) => names.includes(m))) keep.push(e);
  });
  state.game.entries = keep;
  renderCoalitionTable();
  renderGuaranteeTable();
});

// ------------------------------------------------------------- coalition editor
function renderCoalitionTable() {
  const tbody = $("#coalitionTable tbody");
  tbody.innerHTML = "";
  const players = state.game.players;
  const sorted = [...state.game.entries].sort((a, b) =>
    membersToMask(a.members, players) - membersToMask(b.members, players));
  sorted.forEach((entry) => {
    const tr = document.createElement("tr");
    const mask = membersToMask(entry.members, players);

    const tdMembers = document.createElement("td");
    if (mask === 0) {
      tdMembers.textContent = "∅（空联盟，恒为 0）";
    } else {
      const box = document.createElement("div");
      box.className = "row-gap";
      players.forEach((p, i) => {
        const lab = document.createElement("label");
        lab.className = "hint";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.checked = !!((mask >> i) & 1);
        cb.addEventListener("change", () => {
          entry.members = players.filter((_, j) =>
            j === i ? cb.checked : membersToMask(entry.members, players) & (1 << j));
        });
        lab.append(cb, document.createTextNode(p));
        box.appendChild(lab);
      });
      tdMembers.appendChild(box);
    }
    tr.appendChild(tdMembers);

    const mkNum = (key) => {
      const td = document.createElement("td");
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any";
      inp.value = entry[key] === null || entry[key] === undefined ? "" : entry[key];
      inp.placeholder = key === "value" ? "精确值" : (key === "lo" ? "下界" : "上界");
      inp.addEventListener("input", () => {
        entry[key] = inp.value === "" ? null : Number(inp.value);
      });
      td.appendChild(inp);
      return td;
    };
    tr.appendChild(mkNum("value"));
    tr.appendChild(mkNum("lo"));
    tr.appendChild(mkNum("hi"));

    const tdCost = document.createElement("td");
    const costInp = document.createElement("input");
    costInp.type = "number"; costInp.step = "any"; costInp.value = entry.cost ?? 0;
    costInp.addEventListener("input", () => { entry.cost = Number(costInp.value) || 0; });
    tdCost.appendChild(costInp);
    tr.appendChild(tdCost);

    const tdLock = document.createElement("td");
    const lockCb = document.createElement("input");
    lockCb.type = "checkbox"; lockCb.checked = !!entry.locked;
    lockCb.addEventListener("change", () => { entry.locked = lockCb.checked; });
    tdLock.appendChild(lockCb);
    tr.appendChild(tdLock);

    const tdDel = document.createElement("td");
    const del = document.createElement("button");
    del.className = "btn btn-small"; del.textContent = "删除";
    del.disabled = mask === 0;
    del.addEventListener("click", () => {
      state.game.entries = state.game.entries.filter((e) => e !== entry);
      renderCoalitionTable();
    });
    tdDel.appendChild(del);
    tr.appendChild(tdDel);
    tbody.appendChild(tr);
  });
}

$("#addCoalitionBtn").addEventListener("click", () => {
  if (state.game.players.length === 0)
    return showMessages($("#inputMessages"), "请先设置参与方。", "error");
  state.game.entries.push({
    members: [], value: null, lo: null, hi: null, cost: 0, note: "", locked: false,
  });
  renderCoalitionTable();
});

function collectGame() {
  state.game.name = $("#gameName").value || "未命名游戏";
  // ensure empty coalition row exists
  if (!state.game.entries.some((e) => e.members.length === 0)) {
    state.game.entries.unshift({ members: [], value: 0, lo: 0, hi: 0, cost: 0 });
  }
  return state.game;
}

$("#resetBtn").addEventListener("click", () => {
  state.game = { name: "", players: [], entries: [] };
  state.analysis = null;
  $("#gameName").value = ""; $("#playersInput").value = "";
  renderCoalitionTable(); renderGuaranteeTable();
  $("#shapleyChart").innerHTML = ""; $("#unstableTable tbody").innerHTML = "";
  $("#planResult").innerHTML = ""; $("#auditBox").innerHTML = "";
  $("#modeTiles").innerHTML = ""; $("#stabilityVerdict").innerHTML = "";
});

// ------------------------------------------------------------------- analyze
$("#analyzeBtn").addEventListener("click", async () => {
  try {
    const game = collectGame();
    const data = await api("/api/analyze", game);
    const errs = [];
    if (data.shapley && data.shapley.ok === false) errs.push(...(data.shapley.errors || []));
    if (errs.length) {
      showMessages($("#inputMessages"), errs, "error");
      return;
    }
    state.analysis = data;
    renderAnalysis(data);
    showMessages($("#inputMessages"),
      ["分析完成：贡献、稳定性与核对结果已更新。"], "ok");
  } catch (e) {
    showMessages($("#inputMessages"), e.message, "error");
  }
});

function renderAnalysis(data) {
  const { shapley, stability } = data;
  const players = state.game.players;
  renderModeTiles(shapley, stability);
  renderShapleyChart(players, shapley);
  renderStability(players, stability);
  renderAudit(shapley, stability);
}

function renderModeTiles(shapley, stability) {
  const tiles = [
    ["求解模式", ({ exact: "精确枚举", interval: "区间传播", "sampling-required": "需抽样" })[shapley.mode] || shapley.mode],
    ["参与方数", shapley.n],
    ["核心", stability.core_nonempty === true ? "非空"
          : stability.core_nonempty === false ? "为空" : "区间/待定"],
    ["最小核 ε*", stability.epsilon === null || stability.epsilon === undefined
          ? "—" : fmt(stability.epsilon)],
  ];
  $("#modeTiles").innerHTML = tiles.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

// ------------------------------------------------------------- SVG bar chart
function renderShapleyChart(players, shapley) {
  const host = $("#shapleyChart");
  host.innerHTML = "";
  const n = players.length;
  const rowH = 34, labelW = 78, valueW = 150, marginR = 16;
  const width = Math.max(420, host.clientWidth - 4);
  const plotW = width - labelW - valueW - marginR;
  const height = n * rowH + 28;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", width); svg.setAttribute("height", height);
  svg.setAttribute("role", "img");

  const lo = shapley.bounds_lo || [];
  const hi = shapley.bounds_hi || [];
  const point = shapley.exact;
  const finiteVals = [...lo, ...hi, ...(point || [])].filter((v) => typeof v === "number");
  const vMin = Math.min(0, ...finiteVals);
  const vMax = Math.max(0, ...finiteVals, 1e-9);
  const span = vMax - vMin || 1;
  const x = (v) => labelW + ((v - vMin) / span) * plotW;
  const zeroX = x(0);

  // gridlines at 0, min, max
  [vMin, 0, vMax].forEach((g) => {
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", x(g)); line.setAttribute("x2", x(g));
    line.setAttribute("y1", 8); line.setAttribute("y2", height - 18);
    line.setAttribute("class", g === 0 ? "axis-line" : "grid-line");
    svg.appendChild(line);
  });

  players.forEach((p, i) => {
    const y = 12 + i * rowH;
    const color = SERIES[i % SERIES.length];
    // label
    const t = document.createElementNS(svg.namespaceURI, "text");
    t.setAttribute("x", 6); t.setAttribute("y", y + 14);
    t.setAttribute("class", "bar-label"); t.textContent = p;
    svg.appendChild(t);

    const l = lo[i], h = hi[i], pv = point ? point[i] : (l + h) / 2;
    // interval band
    if (typeof l === "number" && typeof h === "number" && Math.abs(h - l) > 1e-12) {
      const band = document.createElementNS(svg.namespaceURI, "rect");
      band.setAttribute("x", x(l)); band.setAttribute("y", y + 6);
      band.setAttribute("width", Math.max(1, x(h) - x(l)));
      band.setAttribute("height", 16); band.setAttribute("class", "interval-band");
      svg.appendChild(band);
    }
    // main bar (point estimate or midpoint)
    if (typeof pv === "number") {
      const bar = document.createElementNS(svg.namespaceURI, "rect");
      const bx = Math.min(zeroX, x(pv));
      bar.setAttribute("x", bx); bar.setAttribute("y", y + 4);
      bar.setAttribute("width", Math.max(2, Math.abs(x(pv) - zeroX)));
      bar.setAttribute("height", 20); bar.setAttribute("fill", color);
      bar.setAttribute("class", "bar");
      svg.appendChild(bar);
      bar.dataset.tip = `${p}：${fmt(pv)}` +
        (typeof l === "number" ? `\n区间 [${fmt(l)}, ${fmt(h)}]` : "");
      bar.style.cursor = "pointer";
    }
    // value text
    const vt = document.createElementNS(svg.namespaceURI, "text");
    vt.setAttribute("x", labelW + plotW + 8); vt.setAttribute("y", y + 15);
    vt.setAttribute("class", "bar-value");
    vt.textContent = point ? fmt(pv)
      : `[${fmt(l)}, ${fmt(h)}]`;
    svg.appendChild(vt);
  });

  const tooltip = ensureTooltip();
  svg.querySelectorAll("rect.bar, rect.interval-band").forEach((el) => {
    el.addEventListener("mousemove", (ev) => {
      tooltip.style.display = "block";
      tooltip.textContent = el.dataset.tip || "区间范围";
      tooltip.style.left = `${ev.clientX + 12}px`;
      tooltip.style.top = `${ev.clientY + 12}px`;
    });
    el.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  });
  host.appendChild(svg);

  $("#shapleyNotes").textContent = shapley.sampling_blocker
    ? "抽样受限：" + shapley.sampling_blocker
    : (point ? "精确 Shapley 值；区间与点值重合。"
             : "条形为界内中点，浅色带为严格上下界传播范围，缺失价值未按零处理。");
}

function ensureTooltip() {
  let tip = document.querySelector(".tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    document.body.appendChild(tip);
  }
  return tip;
}

// ------------------------------------------------------------- stability view
function renderStability(players, st) {
  const verdictEl = $("#stabilityVerdict");
  if (st.scenario === "bounded") {
    verdictEl.className = "verdict unknown";
    verdictEl.textContent = st.verdict || st.message || "区间博弈";
    $("#leastCoreMeta").textContent =
      `最小核 ε* 区间：[${fmt(st.epsilon_interval?.[0])}, ${fmt(st.epsilon_interval?.[1])}]` +
      (st.unbounded_coalitions?.length ? `；无界联盟：${st.unbounded_coalitions.join("、")}` : "");
  } else {
    const nonempty = !!st.core_nonempty;
    verdictEl.className = `verdict ${nonempty ? "nonempty" : "empty"}`;
    verdictEl.innerHTML = nonempty
      ? `<span class="badge good">核心非空</span> 存在同时满足效率与所有联盟理性的分配。`
      : `<span class="badge critical">核心为空</span> 没有任何分配能同时满足所有联盟约束；最小核 ε*=${fmt(st.epsilon)} 是所需的最小统一补偿，绑定联盟 ${st.binding?.length || 0} 个即证据。`;
    $("#leastCoreMeta").textContent =
      `约束生成迭代 ${st.iterations} 次，覆盖联盟约束 ${st.coalition_count} 条` +
      (st.used_known_only ? "（仅已录入联盟，未录入部分无法认证）。" : "（全联盟枚举）。");
  }

  const tbody = $("#unstableTable tbody");
  tbody.innerHTML = "";
  const rows = st.unstable_coalitions || [];
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="hint">暂无联盟排名数据。</td></tr>`;
    return;
  }
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const members = r.members.join(", ");
    tr.innerHTML = `
      <td>{${members}}</td>
      <td class="num">${fmt(r.worth)}</td>
      <td class="num">${fmt(r.allocated)}</td>
      <td class="num">${fmt(r.deficit)}</td>
      <td>${r.blocking
        ? '<span class="badge critical">有退出动机</span>'
        : '<span class="badge good">稳定</span>'}</td>`;
    tbody.appendChild(tr);
  });
}

// ------------------------------------------------------------- guarantees/plan
function renderGuaranteeTable() {
  const tbody = $("#guaranteeTable tbody");
  tbody.innerHTML = "";
  state.guarantees = state.game.players.map((p, i) => state.guarantees[i] || { value: null, locked: false });
  state.game.players.forEach((p, i) => {
    const g = state.guarantees[i];
    const tr = document.createElement("tr");
    const tdName = document.createElement("td"); tdName.textContent = p; tr.appendChild(tdName);
    const tdVal = document.createElement("td");
    const inp = document.createElement("input");
    inp.type = "number"; inp.step = "any"; inp.placeholder = "无保障";
    inp.value = g.value === null ? "" : g.value;
    inp.addEventListener("input", () => { g.value = inp.value === "" ? null : Number(inp.value); });
    tdVal.appendChild(inp); tr.appendChild(tdVal);
    const tdLock = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = g.locked;
    cb.addEventListener("change", () => { g.locked = cb.checked; });
    tdLock.appendChild(cb); tr.appendChild(tdLock);
    tbody.appendChild(tr);
  });
}

$("#planBtn").addEventListener("click", async () => {
  try {
    const game = collectGame();
    // commit is not required for analysis; allow ad-hoc plan via game payload
    const payload = {
      game,
      guarantees: state.guarantees.map((g) => g.locked ? g.value : (g.value)),
      budget: $("#budgetInput").value === "" ? null : Number($("#budgetInput").value),
    };
    const plan = await api("/api/plan", payload);
    renderPlan(plan);
  } catch (e) {
    $("#planResult").innerHTML = `<div class="msg error">${e.message}</div>`;
  }
});

function renderPlan(plan) {
  const players = state.game.players;
  const badge = plan.feasible
    ? '<span class="badge good">可行</span>'
    : '<span class="badge critical">保障不可同时满足</span>';
  let rows = players.map((p, i) => `
    <tr><td>${p}${state.guarantees[i]?.locked ? ' <span class="pill locked">已锁定</span>' : ""}</td>
    <td class="num">${fmt(plan.target[i])}</td>
    <td class="num">${fmt(plan.allocation[i])}</td>
    <td class="num">${fmt(plan.floors[i])}</td>
    <td class="num">${fmt(plan.guarantee_gaps[p] ?? 0)}</td></tr>`).join("");
  let bounded = "";
  if (plan.bounded_plan) {
    const bp = plan.bounded_plan;
    bounded = `<h4 class="sub">界传播下的方案范围</h4>
      <div class="table-wrap"><table><thead><tr><th>参与方</th><th>低目标情景</th><th>高目标情景</th></tr></thead>
      <tbody>${players.map((p, i) => `<tr><td>${p}</td>
        <td class="num">${fmt(bp.allocation_under_low_targets[i])}</td>
        <td class="num">${fmt(bp.allocation_under_high_targets[i])}</td></tr>`).join("")}
      </tbody></table></div>`;
  }
  $("#planResult").innerHTML = `
    <div class="row-between" style="margin:10px 0 6px">${badge}
      <span class="hint">L2 距离 ${fmt(plan.distance_l2)}；预算残差 ${fmt(plan.budget_residual)}</span></div>
    ${plan.feasible ? "" : `<div class="msg error">${plan.note}</div>`}
    <div class="table-wrap"><table>
      <thead><tr><th>参与方</th><th>原贡献目标</th><th>可行分配</th><th>保障下限</th><th>保障缺口</th></tr></thead>
      <tbody>${rows}</tbody></table></div>${bounded}`;
}

// ------------------------------------------------------------------- versions
$("#commitVersionBtn").addEventListener("click", async () => {
  try {
    const game = collectGame();
    const v = await api("/api/versions", {
      game,
      plan: { guarantees: state.guarantees, budget: $("#budgetInput").value || null },
    });
    await refreshVersions();
    showMessages($("#inputMessages"), `已提交版本 ${v.version.id}，更早版本自动转为只读。`, "ok");
  } catch (e) {
    showMessages($("#inputMessages"), e.message, "error");
  }
});

async function refreshVersions() {
  const data = await api("/api/versions");
  const host = $("#versionList");
  host.innerHTML = "";
  if (data.versions.length === 0) {
    host.innerHTML = `<div class="hint">尚无版本。提交后旧版本将只读保留。</div>`;
    return;
  }
  [...data.versions].reverse().forEach((v) => {
    const div = document.createElement("div");
    div.className = "version-item " + (v.read_only ? "readonly" : "");
    div.innerHTML = `
      <div class="row-between">
        <span class="vid">${v.name}</span>
        ${v.read_only ? '<span class="pill">只读</span>' : '<span class="pill locked">当前</span>'}
      </div>
      <div class="meta">${v.id.slice(0, 22)}… · ${v.created_at}</div>
      <div class="meta">${v.players.length} 参与方 · 已存方案 ${v.allocation_count} 个
        ${v.parent_id ? "· 派生自上一版本" : ""}</div>`;
    host.appendChild(div);
  });
}

// ------------------------------------------------------------------- sampling
$("#createTaskBtn").addEventListener("click", async () => {
  try {
    const versions = await api("/api/versions");
    if (!versions.active) throw new Error("请先提交一个版本，再为其创建抽样任务。");
    const task = await api("/api/tasks", {
      version_id: versions.active,
      seed: Number($("#seedInput").value),
      batch_size: Number($("#batchInput").value) || 200,
    });
    state.activeTaskId = task.task.id;
    await refreshTasks();
  } catch (e) {
    showMessages($("#inputMessages"), e.message, "error");
  }
});

$("#advanceBtn").addEventListener("click", async () => {
  try {
    if (!state.activeTaskId) {
      const tasks = await api("/api/tasks");
      if (tasks.tasks.length === 0) throw new Error("请先新建抽样任务。");
      state.activeTaskId = tasks.tasks[0].id;
    }
    const nPerm = Number($("#permInput").value);
    const result = await api(`/api/tasks/${state.activeTaskId}/advance`, { permutations: nPerm });
    await refreshTasks();
    showMessages($("#inputMessages"),
      `本次新增 ${result.new_permutations} 个排列（${result.consumed_before} → ${result.consumed_after}），无重复累计。`,
      "ok");
  } catch (e) {
    showMessages($("#inputMessages"), e.message, "error");
  }
});

async function refreshTasks() {
  const data = await api("/api/tasks");
  const host = $("#taskList");
  host.innerHTML = "";
  if (data.tasks.length === 0) {
    host.innerHTML = `<div class="hint">尚无抽样任务。</div>`;
    return;
  }
  for (const t of [...data.tasks].reverse()) {
    const report = await api(`/api/tasks/${t.id}`);
    if (!state.activeTaskId) state.activeTaskId = t.id;
    const div = document.createElement("div");
    div.className = "task-item";
    const est = report.estimate
      ? report.estimate.map((v, i) => `${state.game.players[i] || "P" + i}=${v === null ? "—" : fmt(v)}`).join("，")
      : "尚未抽样";
    const ci = report.ci95_halfwidth && report.ci95_halfwidth[0] !== null
      ? `95% 半宽 ≈ ${fmt(report.ci95_halfwidth[0])}` : "误差估计待更多批次";
    div.innerHTML = `
      <div class="row-between"><strong>${t.id.slice(0, 18)}…</strong>
        <span class="pill">${report.done} 排列</span></div>
      <div class="meta">种子 ${report.seed ?? t.state?.seed} · 批次 ${report.batch_count} · ${ci}</div>
      <div class="meta">${est}</div>`;
    div.style.cursor = "pointer";
    div.addEventListener("click", () => { state.activeTaskId = t.id; });
    host.appendChild(div);
  }
}

// ---------------------------------------------------------------------- audit
function renderAudit(shapley, stability) {
  const lines = [];
  if (shapley.exact) {
    const gap = shapley.efficiency_gap;
    lines.push({ cls: gap < 1e-8 ? "ok" : "bad",
      text: `效率守恒：ΣShapley 与大联盟净值偏差 = ${fmt(gap, 10)}（应严格为 0）` });
  }
  if (Array.isArray(shapley.bounds_lo)) {
    const ordered = shapley.bounds_lo.every((l, i) =>
      l === null || shapley.bounds_hi[i] === null || l <= shapley.bounds_hi[i] + 1e-10);
    lines.push({ cls: ordered ? "ok" : "bad", text: "区间单调性：每个参与方下界 ≤ 上界" });
    if (shapley.interval_efficiency) {
      const ie = shapley.interval_efficiency;
      lines.push({ cls: "warn",
        text: `区间外包络 Σ∈[${fmt(ie.sum_lo)}, ${fmt(ie.sum_hi)}] 宽于大联盟界 [${fmt(ie.grand_lo)}, ${fmt(ie.grand_hi)}]：独立界之和本就更宽，非计算错误。` });
    }
  }
  if (stability.scenario !== "bounded") {
    if (stability.core_nonempty) {
      lines.push({ cls: "ok", text: "核心非空：返回分配满足全部已枚举联盟约束。" });
    } else if (stability.epsilon !== undefined) {
      lines.push({ cls: "bad",
        text: `核心为空的证据：ε=0 不可行，最小统一补偿 ε*=${fmt(stability.epsilon)}，${stability.binding?.length || 0} 个联盟同时绑定。` });
    }
  } else {
    lines.push({ cls: "warn", text: "区间博弈：结论按界的极端情形分别求解，未声称单一“公平”答案。" });
  }
  if (stability.used_known_only)
    lines.push({ cls: "warn", text: "仅认证已录入联盟；未录入联盟的稳定性无保证。" });
  $("#auditBox").innerHTML = lines.map((l) =>
    `<div class="${l.cls}">${l.text}</div>`).join("");
}

// --------------------------------------------------------------------- theme
const themeToggle = $("#themeToggle");
const savedTheme = localStorage.getItem("theme");
if (savedTheme === "dark") {
  document.documentElement.setAttribute("data-theme", "dark");
  themeToggle.checked = true;
}
themeToggle.addEventListener("change", () => {
  if (themeToggle.checked) {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("theme", "light");
  }
  if (state.analysis) renderAnalysis(state.analysis);
});

// ---------------------------------------------------------------------- boot
(async function boot() {
  try {
    await initExamples();
    const ex = await api(`/api/examples/symmetry`);
    loadGameIntoEditor(ex.game);
    await refreshVersions();
    await refreshTasks();
  } catch (e) {
    showMessages($("#inputMessages"), "初始化失败：" + e.message, "error");
  }
})();
