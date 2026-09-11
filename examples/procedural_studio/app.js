"use strict";

const $ = (id) => document.getElementById(id);
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const names = {
  start: "Start",
  search: "Search",
  read: "Read",
  check: "Reconcile",
  answer: "Review",
  abstain: "Hold",
};
const descriptions = {
  start: "The task begins",
  search: "Retrieve the lot packet",
  read: "Read controlled records",
  check: "Reconcile the evidence",
  answer: "Ready for human review",
  abstain: "Hold for human review",
};
const nodes = {
  start: { x: 50, y: 145, sub: "TASK", symbol: "" },
  search: { x: 170, y: 145, sub: "retrieve lot packet", symbol: "⌕" },
  read: { x: 340, y: 145, sub: "inspect evidence", symbol: "≡" },
  check: { x: 510, y: 240, sub: "scope + exceptions", symbol: "✓" },
  answer: { x: 695, y: 110, sub: "human review", symbol: "↗" },
  abstain: { x: 695, y: 285, sub: "resolve evidence", symbol: "−" },
};
const route = {
  "start:search": { d: "M73 145 H107" },
  "search:read": { d: "M233 145 H277", x: 255, y: 127, label: "read" },
  "read:answer": {
    d: "M403 145 C485 145 537 110 632 110",
    x: 515,
    y: 111,
    label: "skip verification",
  },
  "read:check": {
    d: "M340 178 V218 Q340 240 365 240 H447",
    x: 387,
    y: 260,
    label: "verify first",
  },
  "check:answer": {
    d: "M573 240 H588 Q606 240 606 217 V135 Q606 110 632 110",
    x: 570,
    y: 184,
    label: "complete",
  },
  "check:abstain": {
    d: "M573 240 H588 Q606 240 606 263 V268 Q606 285 632 285",
    x: 594,
    y: 315,
    label: "gap / conflict",
  },
};
const state = {
  data: null,
  phase: 0,
  historical: false,
  caseIndex: 0,
  stepIndex: -1,
  selected: "read",
  playing: false,
  timer: null,
  liveBusy: false,
  liveGeneration: 0,
  liveResult: null,
};
const key = () =>
  state.historical
    ? "historical"
    : ["original", "repaired", "retained"][state.phase];
const currentCase = () => state.data.cases[state.caseIndex];
const run = () => currentCase().runs[key()];
const graph = () => state.data.graphs[key()];
const done = () => state.stepIndex === run().actions.length - 1;

function stop() {
  clearTimeout(state.timer);
  state.timer = null;
  state.playing = false;
}
function reset() {
  stop();
  state.stepIndex = -1;
  state.selected = "read";
  state.liveGeneration++;
  state.liveResult = null;
}
function goPhase(phase) {
  reset();
  state.phase = phase;
  state.historical = false;
  render();
}
function advance() {
  if (done()) reset();
  state.stepIndex++;
  state.liveGeneration++;
  state.liveResult = null;
  state.selected = run().actions[state.stepIndex];
  if (done()) stop();
  render();
}
function tick() {
  if (!state.playing) return;
  advance();
  if (state.playing) state.timer = setTimeout(tick, 850);
}
function togglePlay() {
  if (state.playing) {
    stop();
    renderPlayer();
    renderLive();
    return;
  }
  if (done()) reset();
  state.playing = true;
  tick();
}

function renderGraph() {
  const actual = graph();
  const present = new Set(actual.flatMap((t) => [t.source, t.target]));
  const original = key() === "original" || key() === "historical";
  const active = run().actions[state.stepIndex];
  const visited = run().actions.slice(0, Math.max(0, state.stepIndex));
  const traveled = new Set(
    run()
      .actions.slice(0, Math.max(0, state.stepIndex))
      .map((n, i) => `${n}:${run().actions[i + 1]}`),
  );
  const ghost = original
    ? state.data.graphs.repaired.filter(
        (t) => t.source === "check" || t.target === "check",
      )
    : [];
  let markup = "<defs>";
  for (const [color, id] of [
    ["#899779", "normal"],
    ["#bdc5b5", "ghost"],
    ["#c44b25", "active"],
    ["#527e59", "green"],
    ["#a44636", "red"],
  ])
    markup += `<marker id="arrow-${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="${color}"/></marker>`;
  markup += "</defs>";
  function edge(t, isGhost = false, isRejected = false) {
    const pair = `${t.source}:${t.target}`,
      r = route[pair];
    if (!r) return "";
    const traversed = traveled.has(pair);
    const color = isRejected
      ? "#a44636"
      : isGhost
        ? "#bdc5b5"
        : traversed
          ? "#c44b25"
          : t.source === "check" || t.target === "check"
            ? "#527e59"
            : "#899779";
    const marker = isRejected
      ? "red"
      : isGhost
        ? "ghost"
        : traversed
          ? "active"
          : t.source === "check" || t.target === "check"
            ? "green"
            : "normal";
    return `<g opacity="${isGhost ? 0.65 : 1}"><path class="graph-edge" d="${r.d}" fill="none" stroke="${color}" stroke-width="${traversed ? 2.6 : 1.6}" ${isGhost || isRejected ? 'stroke-dasharray="5 5"' : ""} marker-end="url(#arrow-${marker})"/>${r.label ? `<text class="edge-label" x="${r.x}" y="${r.y}" fill="${color}" text-anchor="middle">${isRejected ? "rejected shortcut" : isGhost && t.target === "check" ? "missing step" : r.label}</text>` : ""}</g>`;
  }
  markup += ghost.map((t) => edge(t, true)).join("");
  markup += actual.map((t) => edge(t)).join("");
  if (state.phase === 2 && !state.historical)
    markup += edge(
      state.data.graphs.original.find((t) => t.key === "finish"),
      false,
      true,
    );
  for (const [id, n] of Object.entries(nodes)) {
    const exists = present.has(id),
      selected = state.selected === id;
    const classes = [
      "graph-node",
      !exists ? "ghost" : "",
      selected ? "selected" : "",
      active === id ? "active" : "",
      visited.includes(id) ? "visited" : "",
    ].join(" ");
    const fill = id === "check" && !original ? "#edf3e6" : "#fffefa";
    const stroke = selected
      ? "#596e48"
      : id === "check" && !original
        ? "#7e9a69"
        : "#c2cbb5";
    markup += `<g class="${classes}" data-node="${id}" role="button" tabindex="0" aria-label="Inspect ${names[id]}${!exists ? " (not in this revision)" : ""}" aria-pressed="${selected}">`;
    if (id === "start") {
      markup += `<circle cx="${n.x}" cy="${n.y}" r="22" fill="${fill}" stroke="${stroke}"/><circle class="pulse" cx="${n.x}" cy="${n.y}" r="5" fill="#758866"/><text x="${n.x}" y="${n.y + 42}" text-anchor="middle" class="node-subtitle" fill="#78816e">START</text>`;
    } else {
      markup += `<rect x="${n.x - 62}" y="${n.y - 32}" width="124" height="64" rx="8" fill="${fill}" stroke="${stroke}" ${!exists ? 'stroke-dasharray="4 4"' : ""}/><text class="node-symbol" x="${n.x - 45}" y="${n.y - 4}" fill="${id === "check" ? "#527e59" : "#849775"}">${n.symbol}</text><text class="node-label" x="${n.x - 22}" y="${n.y - 4}" fill="#303b29">${names[id]}</text><text class="node-subtitle" x="${n.x}" y="${n.y + 17}" text-anchor="middle" fill="#7c8672">${n.sub}</text>`;
      if (id === "check")
        markup += `<text class="node-kicker" x="${n.x}" y="${n.y - 45}" text-anchor="middle" fill="${original ? "#9ba48f" : "#527e59"}">${original ? "NOT YET IN THE GRAPH" : "ADDED BY THE REPAIR"}</text>`;
    }
    markup += "</g>";
  }
  $("graph").innerHTML = markup;
  $("graph-title").textContent = state.historical
    ? "Replay the original belief"
    : [
        "A shortcut through the evidence",
        "Reconcile before the packet advances",
        "A better procedure, protected",
      ][state.phase];
  $("revision").textContent = original ? "REVISION 01" : "REVISION 02";
  $("graph-description").textContent = state.historical
    ? "The old graph still exists, exactly as it was."
    : [
        "A passing report advances the packet without reconciliation.",
        "Check scope and exceptions before routing to human review.",
        "The proposed shortcut never enters the retained graph.",
      ][state.phase];
  $("view-history").disabled = state.phase === 0;
  $("view-history").classList.toggle("active", state.historical);
  $("view-current").classList.toggle("active", !state.historical);
  $("change-legend").innerHTML = original
    ? '<i class="legend-line dashed"></i> Missing check'
    : '<i class="legend-line green"></i> Verified route';
}

function renderPlayer() {
  $("play-icon").textContent = state.playing ? "Ⅱ" : "▶";
  $("play-label").textContent = state.playing
    ? "Pause"
    : done()
      ? "Replay run"
      : "Run the agent";
  $("step").disabled = state.playing;
  $("progress-fill").style.width =
    `${((state.stepIndex + 1) / run().actions.length) * 100}%`;
  $("step-label").textContent =
    state.stepIndex < 0
      ? `Ready to run · ${run().actions.length} steps`
      : `${state.stepIndex + 1} / ${run().actions.length} · ${names[run().actions[state.stepIndex]]}${done() ? " · complete" : ""}`;
}

function renderSources() {
  const c = currentCase(),
    result = run();
  const selected = state.stepIndex >= 0 ? run().actions[state.stepIndex] : null;
  const verified = run()
    .actions.slice(0, state.stepIndex + 1)
    .includes("check");
  $("lot-scope").textContent =
    `${c.lot} · ${c.configuration} · ${c.instruction}`;
  $("sources").innerHTML = c.records
    .map((record, index) => {
      const exception = ["withdrawal", "nonconformance", "test_plan"].includes(
        record.kind,
      );
      const tag =
        record.kind === "withdrawal"
          ? "WITHDRAWAL"
          : record.kind === "test_plan"
            ? "PLAN ONLY"
            : record.status === "open"
              ? "OPEN NCR"
              : record.result === "pass"
                ? "PASS"
                : "APPROVED";
      const chosen =
        verified || ((selected === "read" || done()) && index === 0);
      return `<article class="source ${exception ? "old" : ""} ${chosen ? "chosen" : ""}"><div class="source-top"><span>${escapeHTML(record.record_id)}</span><span class="source-tag">${tag}</span></div><p><strong>${escapeHTML(record.title)}</strong></p><p class="record-narrative">${escapeHTML(record.narrative)}</p>${record.references.length ? `<small class="record-reference">References: ${record.references.map(escapeHTML).join(", ")}</small>` : ""}</article>`;
    })
    .join("");
  $("answer-card").className =
    "answer-card" + (done() ? (result.success ? " success" : " failure") : "");
  if (done()) {
    $("answer").textContent =
      result.answer === "hold_for_review"
        ? "Hold this packet for human review."
        : "Packet ready for human review.";
    $("answer-detail").textContent =
      `${result.success ? "✓ Matches the simulation policy." : "✕ Missed an evidence gap."} ${result.reasons.join(" ")} Cited: ${result.citations.join(", ")}.`;
  } else {
    const text = {
      start: `The agent receives the review request for ${c.lot}.`,
      search: "Retrieving the test report, traveler, and linked exceptions…",
      read: `Reading ${c.records[0].record_id} and its supporting records.`,
      check:
        "Reconciling configuration, instruction revision, withdrawals, and open nonconformances.",
    };
    $("answer").textContent =
      text[selected] || "Run the agent to see its answer.";
    $("answer-detail").textContent = selected
      ? "A step in the recorded execution trace."
      : "Every move follows the selected graph.";
  }
}

function renderGuidance() {
  const outgoing = graph().filter((t) => t.source === state.selected);
  const present = graph().some(
    (t) => t.source === state.selected || t.target === state.selected,
  );
  $("inspector-title").textContent = descriptions[state.selected];
  if (!present) {
    $("guidance").innerHTML =
      '<p class="guidance-empty">This step isn’t part of the original runbook. The repair adds it between reading and answering.</p>';
    return;
  }
  if (!outgoing.length) {
    $("guidance").innerHTML =
      '<p class="guidance-empty">The task ends here. There are no outgoing transitions to follow.</p>';
    return;
  }
  $("guidance").innerHTML = outgoing
    .map((t) => {
      const next = graph().filter((n) => n.source === t.target);
      return `<div class="transition-rule"><div class="rule-path">${names[t.source]} → ${names[t.target]} <span style="opacity:.55">/ hop 1</span></div><div class="rule-label">When · ${escapeHTML(t.condition)}</div><p class="rule-text">${escapeHTML(t.guidance)}</p><p class="rule-pitfall">${escapeHTML(t.pitfalls)}</p>${next.length ? `<div class="rule-label" style="margin-top:14px">Look ahead · hop 2</div><p class="rule-text">${next.map((n) => `${names[n.source]} → ${names[n.target]} (${escapeHTML(n.condition)})`).join("<br>")}</p>` : ""}</div>`;
    })
    .join("");
}

function renderStory() {
  const { accepted, rejected } = state.data;
  const content = [
    {
      symbol: "!",
      eyebrow: "THE FAILURE MODE",
      title: "A green test report isn’t the whole packet.",
      copy: "The shortcut skips configuration, withdrawal, and nonconformance checks. It can advance a packet with unresolved evidence.",
      button: "Validate a repair",
      className: "",
    },
    {
      symbol: "✓",
      eyebrow: `VALIDATION PASSED · ${accepted.before}/${accepted.validation_size} → ${accepted.candidate}/${accepted.validation_size}`,
      title: "Reconcile the evidence before advancing.",
      copy: `The candidate passes all ${accepted.validation_size} validation cases. Anatid stores the new rule and changes its edge in one transaction.`,
      button: "Test a shortcut",
      className: "repaired",
    },
    {
      symbol: "×",
      eyebrow: `CANDIDATE REJECTED · ${rejected.before}/${rejected.validation_size} → ${rejected.candidate}/${rejected.validation_size}`,
      title: "“Skip the check” doesn’t survive validation.",
      copy: "The retained graph stays intact. The failed proposal and its evaluation remain in memory, ready to inform the next refinement.",
      button: state.historical ? "Back to retained" : "Replay original",
      className: "rejected",
    },
  ][state.phase];
  $("story-card").className = `story-card ${content.className}`;
  $("story-symbol").textContent = content.symbol;
  $("story-eyebrow").textContent = content.eyebrow;
  $("story-title").textContent = content.title;
  $("story-copy").textContent = content.copy;
  $("next-chapter").innerHTML = `${content.button} <span>↗</span>`;
}

function renderEvaluation() {
  const before = state.data.evaluation.original,
    after = state.data.evaluation.repaired;
  $("score-before").innerHTML = `${before.passed}<span>/${before.total}</span>`;
  $("score-after").innerHTML =
    state.phase > 0 ? `${after.passed}<span>/${after.total}</span>` : "—";
  $("score-after-note").textContent =
    state.phase > 0 ? "cases passed" : "awaiting repair";
  const labels = state.data.test_labels;
  const verdict = (success) =>
    `<span class="${success ? "pass" : "fail"}">${success ? "✓ Pass" : "× Fail"}</span>`;
  $("test-results").innerHTML = before.outcomes
    .map(
      (o, i) =>
        `<tr><td>${labels[i]}</td><td>${verdict(o.success)}</td><td>${state.phase > 0 ? verdict(after.outcomes[i].success) : '<span class="pending">—</span>'}</td></tr>`,
    )
    .join("");
}

function renderProvenance() {
  const records =
    state.phase === 0 ? state.data.provenance.slice(-1) : state.data.provenance;
  $("provenance").innerHTML =
    records
      .map(
        (record) =>
          `<article class="provenance-item"><small>${escapeHTML(record.time.slice(0, 10))} · ${escapeHTML(record.writer)}</small><h3>${names[record.rule.source]} → ${names[record.rule.target]}</h3><p>${escapeHTML(record.rule.guidance)}</p><details class="raw-evidence"><summary>Inspect stored evidence ↗</summary><pre>${escapeHTML(record.evidence)}</pre></details></article>`,
      )
      .join("") +
    (state.phase === 2
      ? `<article class="provenance-item"><small>2026-09-03 · validation-gate</small><h3>Rejected proposal retained</h3><p>${state.data.negative_memories.length} rejection memory. The current procedure is unchanged.</p><details class="raw-evidence"><summary>Inspect rejected proposal ↗</summary><pre>${escapeHTML(JSON.stringify(state.data.negative_memories[0], null, 2))}</pre></details></article>`
      : "");
}

function modelContext() {
  const c = currentCase();
  const completed = run().actions.slice(0, state.stepIndex + 1);
  const active = completed.at(-1) || "start";
  let frontier = new Set([active]);
  const visited = new Set(frontier),
    neighborhood = [];
  for (let hop = 1; hop <= 2; hop++) {
    const edges = graph().filter((t) => frontier.has(t.source));
    neighborhood.push(...edges.map((t) => ({ hop, ...t })));
    frontier = new Set(
      edges.map((t) => t.target).filter((t) => !visited.has(t)),
    );
    frontier.forEach((t) => visited.add(t));
  }
  return {
    task: c.question,
    simulation_policy: state.data.policy,
    lot: c.lot,
    configuration: c.configuration,
    instruction: c.instruction,
    checkpoint: key(),
    active_procedure: active,
    completed_actions: completed.slice(-6),
    outgoing_neighborhood: neighborhood,
    observed_records: completed.includes("search") ? c.records : [],
  };
}

function renderLive() {
  const config = state.data.llm || {},
    context = modelContext();
  $("llm-badge").textContent = config.enabled
    ? "LIVE AVAILABLE"
    : "OFFLINE REPLAY";
  $("llm-description").textContent = config.enabled
    ? `Ask ${config.model} for guidance at the current replay step. It receives evidence, not the scripted answer or test scores.`
    : "Start with --live to enable OpenRouter. The standalone export supports offline replay only.";
  $("ask-llm").disabled = !config.enabled || state.liveBusy || state.playing;
  $("ask-llm").innerHTML = state.liveBusy
    ? "Asking the model…"
    : 'Ask OpenRouter <span aria-hidden="true">↗</span>';
  $("llm-context").textContent =
    `Active step: ${names[context.active_procedure]} · ${key()}`;
  $("llm-input").textContent = JSON.stringify(context, null, 2);
  $("llm-output").hidden = !state.liveResult;
  if (state.liveResult) {
    const r = state.liveResult;
    if (r.error) {
      $("llm-output").textContent = r.error;
      return;
    }
    const usage = r.usage || {};
    const tokens =
      usage.total_tokens === undefined ? "" : ` · ${usage.total_tokens} tokens`;
    const cost =
      typeof usage.cost === "number" ? ` · $${usage.cost.toFixed(5)}` : "";
    $("llm-output").innerHTML =
      `<span class="eyebrow">MODEL GUIDANCE · ${escapeHTML(r.model)}</span><h3>Next: ${escapeHTML(names[r.next_action] || r.next_action)}</h3><p>${escapeHTML(r.guidance)}</p><p class="llm-citations">${escapeHTML(r.evidence_status.replaceAll("_", " "))} · Sources: ${r.citations.length ? r.citations.map(escapeHTML).join(", ") : "none observed / cited"}</p><small>${(r.elapsed_ms / 1000).toFixed(1)}s${tokens}${cost} · Advice only; no action executed.</small>`;
  }
}

async function askModel() {
  if (state.liveBusy || !state.data.llm.enabled) return;
  stop();
  const generation = state.liveGeneration;
  state.liveBusy = true;
  state.liveResult = null;
  render();
  try {
    const response = await fetch("/api/guide", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Demo-Token": state.data.llm.token,
      },
      body: JSON.stringify({
        case_id: currentCase().name,
        checkpoint: key(),
        step: state.stepIndex,
      }),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.error || "The model request failed.");
    if (generation === state.liveGeneration) state.liveResult = result;
  } catch (error) {
    if (generation === state.liveGeneration)
      state.liveResult = { error: error.message };
  } finally {
    state.liveBusy = false;
    renderLive();
    renderPlayer();
  }
}

function render() {
  document.querySelectorAll(".chapter").forEach((button) => {
    const selected = Number(button.dataset.phase) === state.phase;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-current", selected ? "step" : "false");
  });
  renderGraph();
  renderPlayer();
  renderSources();
  renderGuidance();
  renderStory();
  renderEvaluation();
  renderLive();
}

async function boot() {
  try {
    const embedded = $("demo-data");
    if (embedded) state.data = JSON.parse(embedded.textContent);
    else {
      const response = await fetch("/api/demo");
      if (!response.ok)
        throw new Error(
          `The demo data could not be loaded (${response.status}).`,
        );
      state.data = await response.json();
    }
    $("scenario").innerHTML = state.data.cases
      .map(
        (c, i) =>
          `<option value="${i}">${escapeHTML(c.label)} · ${escapeHTML(c.lot)}</option>`,
      )
      .join("");
    $("scenario").addEventListener("change", () => {
      reset();
      state.caseIndex = Number($("scenario").value);
      render();
    });
    document.querySelectorAll("[data-phase]").forEach((b) =>
      b.addEventListener("click", () => {
        goPhase(Number(b.dataset.phase));
        renderProvenance();
      }),
    );
    $("play").addEventListener("click", togglePlay);
    $("ask-llm").addEventListener("click", askModel);
    $("step").addEventListener("click", advance);
    $("restart").addEventListener("click", () => {
      reset();
      render();
    });
    $("view-current").addEventListener("click", () => {
      reset();
      state.historical = false;
      render();
    });
    $("view-history").addEventListener("click", () => {
      reset();
      state.historical = true;
      render();
    });
    $("next-chapter").addEventListener("click", () => {
      if (state.phase < 2) goPhase(state.phase + 1);
      else {
        reset();
        state.historical = !state.historical;
        render();
      }
      renderProvenance();
    });
    function inspect(event) {
      const target = event.target.closest("[data-node]");
      if (!target) return;
      if (
        event.type === "keydown" &&
        event.key !== "Enter" &&
        event.key !== " "
      )
        return;
      if (event.type === "keydown") event.preventDefault();
      const retainFocus = event.type === "keydown";
      state.selected = target.dataset.node;
      renderGraph();
      renderGuidance();
      if (retainFocus)
        $("graph").querySelector(`[data-node="${state.selected}"]`).focus();
    }
    $("graph").addEventListener("click", inspect);
    $("graph").addEventListener("keydown", inspect);
    $("flat-results").innerHTML = state.data.flat
      .map(
        (t) =>
          `<span class="flat-rule">${names[t.source]} → ${names[t.target]}</span>`,
      )
      .join("");
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        stop();
        renderPlayer();
        renderLive();
      }
    });
    render();
    renderProvenance();
    $("loading").hidden = true;
    $("app").hidden = false;
  } catch (error) {
    $("loading").hidden = true;
    $("error").hidden = false;
    $("error").textContent =
      `Unable to load the demo. ${error.message} Start it with: python -m examples.procedural_studio`;
  }
}
boot();
