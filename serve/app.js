/* ───────────────────────────────────────────────────────────────────────────
   ARENA — a Preact app on a stdlib Python backend, fully offline.
   The view is components; the hard-won logic (the streaming parser that knows
   about reasoning_content, the syntax highlighter that survives a half-arrived
   docstring, the markdown renderer) is kept verbatim from the version that
   earned it. Nothing here reaches a network but the box's own /api.
   ─────────────────────────────────────────────────────────────────────────── */
const { h, render: mount } = preact;
const { useState, useEffect, useRef, useCallback } = preactHooks;
const html = htm.bind(h);

marked.setOptions({ breaks: true, gfm: true });

const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const DOT = "  ·  ";
const cid = () => "c" + Math.random().toString(36).slice(2, 10);
const shortName = f => (f || "").replace(/-\d{5}-of-\d{5}\.gguf$/, "").replace(/\.gguf$/, "");

async function api(path, opts) {
  const r = await fetch(path, opts);
  const t = await r.text();
  try { return JSON.parse(t); } catch (_) { return { error: t }; }
}

/* ── the syntax highlighter, verbatim ── */
const KW = {
  python: /\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|lambda|None|True|False|yield|assert|pass|break|continue|global|await|async)\b/g,
  javascript: /\b(function|const|let|var|return|if|else|for|while|of|in|new|class|extends|import|from|export|async|await|try|catch|finally|throw|typeof|null|undefined|true|false)\b/g,
  bash: /\b(if|then|else|fi|for|do|done|while|case|esac|function|echo|export|cd|source|return|local)\b/g
};
function hl(code, lang) {
  const parts = [];
  const MARK = "\u0000";
  let s = code.replace(/("""|''')[\s\S]*?\1|("""|''')[\s\S]*$|("|')(?:\\.|(?!\3)[^\\\n])*\3/g,
                       m => MARK + (parts.push(["s", m]) - 1) + MARK)
              .replace(/#[^\n]*|\/\/[^\n]*|\/\*[\s\S]*?\*\//g,
                       m => MARK + (parts.push(["c", m]) - 1) + MARK);
  s = esc(s);
  const kw = KW[(lang || "").toLowerCase()] || KW.python;
  s = s.replace(kw, '<span class="k">$1</span>')
       .replace(/\b(\d+\.?\d*)\b/g, '<span class="n">$1</span>')
       .replace(/\b([A-Za-z_]\w*)(?=\()/g, '<span class="f">$1</span>');
  s = s.replace(new RegExp(MARK + '(?:<span class="n">)?(\\d+)(?:<\\/span>)?' + MARK, "g"),
                (_, i) => { const p = parts[+i]; return p ? '<span class="' + p[0] + '">' + esc(p[1]) + "</span>" : ""; });
  return s;
}
function renderMd(md) {
  let out = marked.parse(md || "");
  out = out.replace(/<pre><code(?: class="language-([^"]*)")?>([\s\S]*?)<\/code><\/pre>/g,
    (_, lang, bodyHtml) => {
      const raw = bodyHtml.replace(/&lt;/g, "<").replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");
      const named = lang
        || (/^\s*(?:def |class |import |from \w+ import |with |print\(|@\w|async def )/m.test(raw) ? "python"
         :  /^\s*(?:function |const |let |var )/m.test(raw)           ? "javascript"
         :  /^\s*(?:#!\/bin\/|sudo |apt |uv |git |cd |curl )/m.test(raw) ? "bash"
         :  "code");
      return '<pre><div class="hd"><span>' + named + '</span>'
        + '<button class="cp" data-code="' + encodeURIComponent(raw) + '">copy</button></div>'
        + "<code>" + hl(raw, named) + "</code></pre>";
    });
  return out;
}
/* copy buttons inside rendered code — one delegated listener, Preact-safe */
document.addEventListener("click", e => {
  const b = e.target.closest && e.target.closest(".cp");
  if (!b) return;
  navigator.clipboard.writeText(decodeURIComponent(b.dataset.code));
  const was = b.textContent; b.textContent = "copied";
  setTimeout(() => { b.textContent = was; }, 1200);
});

/* ── icons (Lucide) — data is a recursive node [tag, attrs, children] ── */
const ICONS = (typeof lucide !== "undefined" && lucide.icons) || {};
function iconNode(spec) {
  if (typeof spec === "string") return spec;
  const [tag, attrs, children] = spec;
  return h(tag, attrs || {}, (children || []).map(iconNode));
}
function Icon({ n, size }) {
  const data = ICONS[n];
  if (!data || !Array.isArray(data)) return html`<svg width=${size || 14} height=${size || 14}></svg>`;
  const attrs = data[1] || {}, children = data[2] || [];
  return h("svg", { ...attrs, width: size || 14, height: size || 14, stroke: "currentColor" },
           children.map(iconNode));
}

/* dangerously-set-html helper for rendered markdown */
const Html = ({ cls, s }) => html`<div class=${cls} dangerouslySetInnerHTML=${{ __html: s }}></div>`;

/* ═══════════════════════════ components ═══════════════════════════ */

function TopBar(p) {
  const { models, sel, setSel, cards, setCards, status, onLoad, onUnload,
          engList, engine, setEngine, engOK, engWhy, toggle } = p;
  const live = [], base = [];
  models.forEach(m => (m.baseline ? base : live).push(m));
  const cur = models.find(m => m.file === sel) || {};
  const st = status || {};
  const loadedMatch = st.ready && st.model === sel && st.cards === cards;

  const opt = m => html`<option value=${m.file} disabled=${m.partial}>
    ${shortName(m.file)}${m.partial ? DOT + "still downloading (" + m.gb + " GB)"
      : DOT + m.gb + " GB" + (m.moe && m.active_b ? DOT + m.active_b + "B active of " + m.total_b + "B"
        : m.active_b ? DOT + m.active_b + "B" : "")}</option>`;

  let runCls = "status", runTxt = st.note || "no model", runLed = "";
  if (st.loading) { runCls = "status busy"; runTxt = "loading …"; }
  else if (st.ready) { runCls = "status on"; runTxt = /—|spill/.test(st.note) ? st.note : "ready"; }
  else if (st.note) { runCls = "status warn"; }

  return html`
    <div class="top">
      <button class="drawer-btn nav" onClick=${() => toggle("nav")} title="Conversations"><${Icon} n="PanelLeft" size=${17}/></button>
      <div class="brand">
        <div class="mark"><${Icon} n="Gauge" size=${16}/></div>
        <div class="name">arena<b>·</b></div>
      </div>

      <div class="controls">
        <select class="sel" value=${sel} onChange=${e => setSel(e.target.value)} title="Model">
          ${live.map(opt)}
          ${base.length ? html`<optgroup label="v1.0 baselines — kept reproducible">${base.map(opt)}</optgroup>` : null}
        </select>

        <select class="sel eng" value=${engine} onChange=${e => setEngine(e.target.value)} title="Serving engine">
          ${(engList || []).map(e => html`<option value=${e.id} disabled=${!engOK(e.id)}>${e.label}${engOK(e.id) ? "" : " — HF only"}</option>`)}
        </select>

        <div class="seglabel">
          <div class="seg" title="Graphics cards to spread the model over">
            ${[1, 2, 3].map(n => html`<button class=${(n === cards ? "on " : "") + (st.ready && st.cards === n ? "live" : "")}
              onClick=${() => setCards(n)}>${n}</button>`)}
          </div>
          <span class="q" title="How many of the three cards to spread the model over. One is simplest; two or three split it by layer, the only way anything over 8 GB runs here.">?</span>
        </div>

        <div class="btns">
          <button class="btn primary" disabled=${st.loading || !engOK(engine)} onClick=${onLoad} title=${engOK(engine) ? "" : engWhy(engine)}>
            <${Icon} n=${loadedMatch ? "RotateCw" : "Play"} size=${14}/>${loadedMatch ? "Reload" : "Load"}</button>
          <button class="btn" disabled=${!st.ready || st.loading} onClick=${onUnload}>
            <${Icon} n="Power" size=${14}/>Unload</button>
        </div>
      </div>

      <div class="tele">
        <div class=${runCls} title=${runTxt}><span class="led"></span>${runTxt}</div>
        <div class="meters" title="graphics memory per card">
          ${(st.vram || []).map((c, i) => {
            const pct = c.total ? c.used / c.total * 100 : 0;
            return html`<div class=${"gm" + (pct > 90 ? " hot" : "")}>
              <span class="lb">C${i}</span>
              <i><u style=${"width:" + pct.toFixed(1) + "%"}></u></i>
              <b>${pct.toFixed(0)}<i class="pct">%</i></b>
            </div>`;
          })}
        </div>
      </div>
      <button class="drawer-btn panel" onClick=${() => toggle("panel")} title="Inference & battle"><${Icon} n="PanelRight" size=${17}/></button>
    </div>`;
}

function PlanLine({ models, sel, cards, ready, engine, engOK, engWhy }) {
  const m = models.find(x => x.file === sel);
  if (!m) return html`<div class="plan"><span class="dim">No models in models/gguf.</span></div>`;
  const holds = 7.2 * cards, need = Math.max(1, Math.ceil(m.gb / 7.2));
  const bits = [];
  bits.push(html`<span><b>${shortName(m.file)}</b> is ${m.gb} GB</span>`);
  if (m.partial) bits.push(html`<span class="bad">still downloading — cannot load yet</span>`);
  else if (m.gb > holds) bits.push(html`<span class="bad">${cards} card${cards > 1 ? "s" : ""} hold ~${holds.toFixed(1)} GB, so layers spill to host memory — needs ${need} card${need > 1 ? "s" : ""}</span>`);
  else bits.push(html`<span>fits on ${cards} card${cards > 1 ? "s" : ""}, ~${(holds - m.gb).toFixed(1)} GB left for cache</span>`);
  if (cards > 1) bits.push(html`<span class="dim">layer split — measured bit-identical to one card</span>`);
  if (m.moe) bits.push(html`<span class="dim">mixture of experts — a fraction of the weights per token</span>`);
  if (engine && engOK && !engOK(engine)) bits.push(html`<span class="bad">${engine}: ${engWhy(engine)}</span>`);
  else if (engine && engine !== "llama.cpp") bits.push(html`<span class="dim">engine: ${engine}</span>`);
  if (ready) bits.push(html`<span class="dim">loading stops what runs first: llama.cpp can't release weights in place</span>`);
  const out = [];
  bits.forEach((b, i) => { if (i) out.push(html`<span class="sep">/</span>`); out.push(b); });
  return html`<div class="plan">${out}</div>`;
}

function Sidebar({ convs, activeId, onOpen, onNew, onDel }) {
  return html`
    <div class="col">
      <div class="colhead">
        <span>Conversations</span>
        <button class="newbtn" onClick=${onNew}><${Icon} n="Plus" size=${12}/>new</button>
      </div>
      <div class="scroll">
        ${convs.length ? convs.map(c => html`
          <div class=${"cv" + (c.id === activeId ? " on" : "")} onClick=${() => onOpen(c.id)}>
            <button class="x" onClick=${e => { e.stopPropagation(); onDel(c.id); }}><${Icon} n="Trash2" size=${12}/></button>
            <span class="t">${c.title || "untitled"}</span>
            <span class="m">${c.n} msgs${DOT}${(c.model || "").replace(/\.gguf$/, "").slice(0, 22)}</span>
          </div>`)
          : html`<p class="hint" style="padding:10px 6px">No conversations yet.</p>`}
      </div>
    </div>`;
}

function Message({ m, i, onCopy, onEdit, onRegen, onMore }) {
  const isUser = m.role === "user";
  const cut = m.finish === "length"
    ? html`<div class="cut"><${Icon} n="TriangleAlert" size=${13}/><span>cut off at the ${m.cap || "token"}-token limit — not the end of the answer</span>
        <button class="more" onClick=${() => onMore(i)}>Continue</button></div>`
    : m.finish === "stopped" ? html`<div class="cut"><span>stopped</span></div>` : null;
  return html`
    <div class=${"m " + (isUser ? "u" : "a")}>
      <div class="who">
        <span class="badge"><${Icon} n=${isUser ? "User" : "Sparkles"} size=${12}/>${isUser ? "you" : "model"}</span>
        <span class="acts">
          ${isUser
            ? html`<button class="act" onClick=${() => onEdit(i)}><${Icon} n="Pencil" size=${13}/></button>`
            : html`<button class="act" onClick=${() => onRegen(i)}><${Icon} n="RotateCw" size=${13}/></button>
                   <button class="act" onClick=${() => onCopy(m.content)}><${Icon} n="Copy" size=${13}/></button>`}
        </span>
      </div>
      ${isUser
        ? html`<div class="bub">${m.content}</div>`
        : html`<div class="bub">
            ${m.think ? html`<details class="think"><summary>thinking · ${m.think.length.toLocaleString()} characters</summary><div>${m.think}</div></details>` : null}
            <${Html} cls=${"md" + (m.streaming && !m.content ? " typing" : "")} s=${renderMd(m.content)}/>
            ${m.streaming && m.content ? html`<span class="typing"></span>` : null}
            ${cut}
          </div>`}
    </div>`;
}

function Chat(p) {
  const { conv, status, cap, setCap, temp, setTemp, input, setInput, busy, speed,
          onSend, onStop, onCopy, onEdit, onRegen, onMore } = p;
  const msgs = (conv && conv.messages) || [];
  const scroller = useRef(null);
  useEffect(() => { if (scroller.current) scroller.current.scrollTop = 1e9; }, [msgs.length, busy]);
  const st = status || {};
  const model = shortName(st.model);

  const empty = st.ready
    ? html`<div class="empty"><div class="orb"><${Icon} n="Sparkles" size=${30}/></div>
        <h2>${model || "Ready"}</h2>
        <p>Loaded across ${st.cards} card${st.cards > 1 ? "s" : ""}. Ask it to write something, fix a bug, or explain a file — the speed and token count show as it answers.</p></div>`
    : html`<div class="empty off"><div class="orb"><${Icon} n="Power" size=${30}/></div>
        <h2>No model loaded</h2>
        <p>Pick a model and a card count up top, then Load. The line below the bar says what that will do before it does it.</p></div>`;

  return html`
    <div class="col">
      <div class="msgs" ref=${scroller}>
        ${msgs.length ? msgs.map((m, i) => html`<${Message} m=${m} i=${i} key=${i}
            onCopy=${onCopy} onEdit=${onEdit} onRegen=${onRegen} onMore=${onMore}/>`) : empty}
      </div>
      <div class="comp">
        <div class="box">
          <textarea placeholder="Ask anything.  Enter to send, Shift+Enter for a new line." value=${input}
            onInput=${e => setInput(e.target.value)}
            onKeyDown=${e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); } }}></textarea>
          ${busy
            ? html`<button class="btn send" onClick=${onStop}><${Icon} n="Square" size=${13}/>Stop</button>`
            : html`<button class="btn primary send" onClick=${onSend}><${Icon} n="Send" size=${13}/>Send</button>`}
        </div>
        <div class="meta">
          <span>max tokens
            <select value=${cap} onChange=${e => setCap(+e.target.value)}>
              ${[512, 2048, 4096, 8192, 16384].map(n => html`<option value=${n}>${n}</option>`)}
            </select></span>
          <span>temp
            <select value=${temp} onChange=${e => setTemp(+e.target.value)}>
              ${[0, 0.2, 0.5, 0.7, 1].map(n => html`<option value=${n}>${n}</option>`)}
            </select></span>
          ${speed ? html`<span class="spd">${speed}</span>` : null}
        </div>
      </div>
    </div>`;
}

/* ── right panel ── */
function Bench({ bench, ready, onRun }) {
  const b = bench || {};
  let body;
  if (b.error) body = html`<div class="bx"><h4>The benchmark failed</h4><div class="roof">
      <div class="say" style="color:var(--amber)">${b.error}</div>
      <div class="caveat">Nothing was measured. An empty result and a failed run look identical unless one says so.</div></div></div>`;
  else if (b.running) body = html`<div class="bx"><h4>Measuring</h4><div class="roof">
      <div class="cap">${b.phase || "starting"} …</div>
      <div class="bar"><u style="width:100%;opacity:.5"></u></div>
      <div class="caveat">Nothing else should run while this is measured — chat and battle share the engine.</div></div></div>`;
  else if (b.result) {
    const r = b.result, k = r.bottleneck;
    const pct = k ? Math.min(100, k.pct_of_roof) : 0;
    body = html`
      <div class="bx"><h4>Per prompt · median of three</h4>
        <table><tr><th>prompt</th><th>in</th><th>out</th><th>think</th><th>first</th><th>decode</th></tr>
        ${r.prompts.map(p => p.note
          ? html`<tr><td>${p.id}</td><td colspan="5" style="text-align:left;color:var(--faint)">${p.note}</td></tr>`
          : html`<tr><td>${p.id}</td><td>${p.in_tokens || "?"}</td><td>${p.out_tokens}</td>
              <td>${p.think_tokens != null ? p.think_tokens : "—"}</td><td>${p.ttft_ms} ms</td>
              <td class="num">${p.decode_tps}/s</td></tr>`)}
        </table></div>
      <div class="bx"><h4>Streams at once</h4>
        <table><tr><th>streams</th><th>total</th><th>each</th><th>wall</th></tr>
        ${r.concurrency.map(c => html`<tr><td>${c.n}</td><td class="num">${c.total_tps}/s</td><td>${c.per_stream_tps}/s</td><td>${c.wall_s}s</td></tr>`)}
        </table></div>
      <div class="bx"><h4>Cards during the run</h4>
        ${r.gpus.map(g => html`<div class="ubar"><span class="lab">Card ${g.card}</span>
          <i><u style=${"width:" + g.util_mean + "%"}></u></i><span>${g.util_mean}%</span>
          <span style="color:var(--faint)">${(g.peak_mb / 1024).toFixed(1)} GB</span></div>`)}
      </div>
      ${k ? html`<div class="bx"><h4>What is holding it back</h4><div class="roof">
        <span class="big">${k.pct_of_roof}%</span><span class="cap"> of ${r.cards > 1 ? "one card's" : "the card's"} bandwidth</span>
        <div class=${"bar" + (pct >= 55 ? " hot" : "")}><u style=${"width:" + pct + "%"}></u></div>
        <div class="cap">${k.achieved_gbps} GB/s reached · ${k.roof_gbps} rated · reads ${k.reads_per_token}/token</div>
        <div class="say">${k.verdict}</div>
        ${(k.notes || []).map(n => html`<div class="caveat">${n}</div>`)}
      </div></div>` : null}
      <p class="hint" style="margin-top:11px">${r.model.replace(/\.gguf$/, "")}${DOT}${r.cards} card${r.cards > 1 ? "s" : ""}${DOT}${r.when}</p>`;
  }
  return html`<div class="scroll">
    <button class="btn primary" style="width:100%;justify-content:center;margin-top:12px" disabled=${b.running || !ready} onClick=${onRun}>
      <${Icon} n="Gauge" size=${14}/>${b.running ? "Measuring…" : b.result ? "Measure again" : "Measure this model"}</button>
    <p class="hint" style="margin-top:9px">Four frozen prompts, three runs each, then 1/4/16 streams at once. Cards sampled throughout.</p>
    ${body}
  </div>`;
}

function Battle({ battle, board, ready, onRun }) {
  const bt = battle || {};
  const rows = Object.entries(board || {}).sort((a, b) => (b[1].solved - a[1].solved) || (a[1].steps - b[1].steps));
  return html`<div class="scroll">
    <button class="btn primary" style="width:100%;justify-content:center;margin-top:12px" disabled=${bt.running || !ready} onClick=${onRun}>
      <${Icon} n="Swords" size=${14}/>${bt.running ? "Running…" : "Run the battle"}</button>
    ${(bt.log || []).map(t => {
      const cls = t.state === "solved" ? "ok" : t.state === "failed" ? "no" : "run";
      const ic = t.state === "solved" ? "Check" : t.state === "failed" ? "X" : "Loader";
      return html`<div class=${"tk " + cls} style="margin-top:8px">
        <span>${t.task}</span>
        <span class="st"><${Icon} n=${ic} size=${12}/>${t.state}${t.steps ? " · " + t.steps + "st" : ""}</span></div>`;
    })}
    <div class="bx" style="margin-top:14px"><h4>Leaderboard</h4>
      <table class="bd" style="margin:0;padding:8px"><tr><th style="padding:8px 8px 5px">model</th><th>solved</th><th>steps</th><th>tokens</th></tr>
      ${rows.length ? rows.map(([k, v]) => html`<tr><td style="padding-left:8px">${k}</td><td class="num">${v.solved}/${v.of}</td><td>${v.steps}</td><td>${(v.sent || 0).toLocaleString()}</td></tr>`)
        : html`<tr><td colspan="4" class="hint" style="padding:8px">nothing yet</td></tr>`}
      </table></div>
    <p class="hint" style="margin-top:12px">Solved means the test command exited 0. The model may write source files; it is refused permission to edit tests.</p>
  </div>`;
}

function RightPanel(p) {
  const [tab, setTab] = useState("speed");
  return html`
    <div class=${"col" + (p.open ? " open" : "")}>
      <div class="tabs">
        <button class=${"tab" + (tab === "speed" ? " on" : "")} onClick=${() => setTab("speed")}><${Icon} n="Activity" size=${13}/>Inference</button>
        <button class=${"tab" + (tab === "battle" ? " on" : "")} onClick=${() => setTab("battle")}><${Icon} n="Swords" size=${13}/>Test battle</button>
      </div>
      ${tab === "speed"
        ? html`<${Bench} bench=${p.bench} ready=${p.ready} onRun=${p.onBench}/>`
        : html`<${Battle} battle=${p.battle} board=${p.board} ready=${p.ready} onRun=${p.onBattle}/>`}
    </div>`;
}

/* ═══════════════════════════ the app ═══════════════════════════ */
function App() {
  const [models, setModels] = useState([]);
  const [status, setStatus] = useState(null);
  const [sel, setSel] = useState("");
  const [cards, setCards] = useState(1);
  const [engine, setEngine] = useState("llama.cpp");
  const [convs, setConvs] = useState([]);
  const [conv, setConv] = useState({ id: cid(), title: "", model: "", messages: [] });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState("");
  const [cap, setCap] = useState(2048);
  const [temp, setTemp] = useState(0.2);
  const [board, setBoard] = useState({});
  const [drawer, setDrawer] = useState(null); // 'nav' | 'panel' | null
  const abortRef = useRef(null);
  const convRef = useRef(conv);
  convRef.current = conv;
  const ready = !!(status && status.ready && !status.loading);
  const [engList, setEngList] = useState([{id:"llama.cpp",label:"llama.cpp"}]);
  const selModel = models.find(m => m.file === sel);
  const selHF = !!(selModel && selModel.hf);
  const engOK = id => id === "llama.cpp" ? !selHF : selHF;
  const engWhy = id => engOK(id) ? "" : (id === "llama.cpp"
    ? "llama.cpp serves GGUF, not safetensors — pick a .gguf model"
    : id === "sglang"
    ? "SGLang needs an HF (safetensors) model, not a bare .gguf"
    : "this vLLM build needs an HF (safetensors) model, not a .gguf");

  const remember = (m, c) => { try { localStorage.setItem("arena", JSON.stringify({ model: m, cards: c })); } catch (_) {} };

  const pickCards = useCallback((n, m) => {
    const nn = Math.max(1, Math.min(3, n || 1));
    setCards(nn); remember(m != null ? m : sel, nn);
  }, [sel]);

  const chooseModel = f => {
    setSel(f);
    const m = models.find(x => x.file === f);
    let c = cards;
    if (m && m.gb > 7.2 * cards) { c = Math.ceil(m.gb / 7.2); setCards(c); }
    remember(f, c);
  };

  /* boot */
  useEffect(() => { (async () => {
    const [ms, st, egs] = await Promise.all([api("/api/models"), api("/api/status"), api("/api/engines")]);
    if (Array.isArray(egs) && egs.length) setEngList(egs);
    setModels(ms); setStatus(st);
    let saved = {}; try { saved = JSON.parse(localStorage.getItem("arena") || "{}"); } catch (_) {}
    const usable = f => f && ms.some(m => m.file === f && !m.partial);
    const start = usable(st && st.model) ? st.model : usable(saved.model) ? saved.model
      : (ms.find(m => !m.partial && !m.baseline) || ms.find(m => !m.partial) || {}).file;
    if (start) setSel(start);
    setCards(Math.max(1, Math.min(3, (st && st.ready && st.cards) || saved.cards || 1)));
    listConvs(); loadBoard();
  })(); }, []);

  /* poll */
  useEffect(() => {
    const t = setInterval(async () => {
      const s = await api("/api/status");
      setStatus(s);
      if (s.battle && !s.battle.running && document.__fighting) { document.__fighting = false; loadBoard(); }
    }, 1500);
    return () => clearInterval(t);
  }, []);

  async function listConvs() { setConvs(await api("/api/conversations")); }
  async function loadBoard() { setBoard(await api("/api/board")); }

  async function saveConv(c) {
    if (!c.messages.length) return;
    c.model = sel;
    if (!c.title) c.title = (c.messages[0] && c.messages[0].content || "").slice(0, 60);
    await api("/api/conversation", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) });
    listConvs();
  }

  async function openConv(id) { const d = await api("/api/conversation/" + id); if (d && !d.error) setConv(d); }
  function newConv() { setConv({ id: cid(), title: "", model: sel, messages: [] }); }
  async function delConv(id) {
    await fetch("/api/conversation/" + id, { method: "DELETE" });
    if (convRef.current.id === id) newConv();
    listConvs();
  }

  /* the streaming turn — reasoning_content and content both counted */
  async function stream(base) {
    if (busy) return;
    setBusy(true); setSpeed("");
    const ac = new AbortController(); abortRef.current = ac;
    let acc = "", think = "", finish = null, n = 0, first = null;
    const t0 = performance.now();
    const draft = { ...base, messages: [...base.messages, { role: "assistant", content: "", think: "", streaming: true }] };
    setConv(draft);
    const idx = draft.messages.length - 1;
    const paint = () => {
      const msgs = draft.messages.slice();
      msgs[idx] = { ...msgs[idx], content: acc, think };
      draft.messages = msgs;
      setConv({ ...draft, messages: msgs });
    };
    try {
      const res = await fetch("/api/chat", { method: "POST", signal: ac.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: base.messages, max_tokens: cap, temperature: temp }) });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || "request failed"); }
      const rd = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const r2 = await rd.read(); if (r2.done) break;
        buf += dec.decode(r2.value, { stream: true });
        const ps = buf.split("\n\n"); buf = ps.pop();
        for (const pk of ps) {
          const l = pk.trim(); if (!l.startsWith("data:")) continue;
          const d = l.slice(5).trim(); if (d === "[DONE]") continue;
          try {
            const o = JSON.parse(d), c = o.choices && o.choices[0], dl = (c && c.delta) || {};
            if (dl.reasoning_content) { if (first === null) first = performance.now(); n++; think += dl.reasoning_content; }
            if (dl.content) { if (first === null) first = performance.now(); n++; acc += dl.content; }
            if (dl.reasoning_content || dl.content) paint();
            if (c && c.finish_reason) finish = c.finish_reason;
          } catch (_) {}
        }
      }
    } catch (e) {
      if (e.name === "AbortError") finish = "stopped";
      else { draft.messages[idx] = { role: "assistant", content: "⚠ " + e.message }; setConv({ ...draft }); setBusy(false); return; }
    }
    const done = { ...base, messages: [...base.messages,
      { role: "assistant", content: acc, think: think || undefined, finish, cap }] };
    setConv(done); saveConv(done);
    setBusy(false); abortRef.current = null;
    const b = first === null ? t0 : first, dt = (performance.now() - b) / 1000;
    setSpeed(n ? n + " chunks · " + (n / Math.max(dt, .001)).toFixed(1) + "/s · first " + (b - t0).toFixed(0) + " ms" : "");
  }

  function send() {
    const v = input.trim(); if (!v || busy) return;
    setInput("");
    stream({ ...convRef.current, messages: [...convRef.current.messages, { role: "user", content: v }] });
  }
  function stop() { if (abortRef.current) abortRef.current.abort(); }
  function copyMsg(t) { navigator.clipboard.writeText(t); }
  function editMsg(i) { const c = convRef.current; setInput(c.messages[i].content); setConv({ ...c, messages: c.messages.slice(0, i) }); }
  function regen(i) { const c = convRef.current; stream({ ...c, messages: c.messages.slice(0, i) }); }
  function more(i) {
    const c = convRef.current; const msgs = c.messages.slice();
    if (msgs[i]) delete msgs[i].finish;
    stream({ ...c, messages: [...msgs, { role: "user", content: "Continue from exactly where you stopped. Do not repeat anything already written." }] });
  }

  async function doLoad() {
    setStatus({ ...status, loading: true });
    const r = await api("/api/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: sel, cards, engine }) });
    remember(sel, cards);
    if (!r.ok) setStatus(s => ({ ...s, loading: false, ready: false, note: String(r.message).slice(0, 90) }));
  }
  function doUnload() { fetch("/api/unload", { method: "POST" }); }
  async function runBench() { const r = await api("/api/bench", { method: "POST" }); if (r.error) setStatus(s => ({ ...s, bench: { ...(s && s.bench), error: r.error } })); }
  async function runBattle() { const r = await api("/api/battle", { method: "POST" }); if (!r.error) document.__fighting = true; }

  return html`
    <${TopBar} models=${models} sel=${sel} setSel=${chooseModel} cards=${cards} setCards=${n => pickCards(n)}
      engList=${engList} engine=${engine} setEngine=${setEngine} engOK=${engOK} engWhy=${engWhy}
      status=${status} onLoad=${doLoad} onUnload=${doUnload} toggle=${k => setDrawer(d => d === k ? null : k)}/>
    <${PlanLine} models=${models} sel=${sel} cards=${cards} ready=${ready} engine=${engine} engOK=${engOK} engWhy=${engWhy}/>
    <div class=${"backdrop" + (drawer ? " show" : "")} onClick=${() => setDrawer(null)}></div>
    <div class="main">
      <${Sidebar} convs=${convs} activeId=${conv.id} onOpen=${id => { openConv(id); setDrawer(null); }} onNew=${() => { newConv(); setDrawer(null); }} onDel=${delConv} open=${drawer === "nav"}/>
      <${Chat} conv=${conv} status=${status} cap=${cap} setCap=${setCap} temp=${temp} setTemp=${setTemp}
        input=${input} setInput=${setInput} busy=${busy} speed=${speed}
        onSend=${send} onStop=${stop} onCopy=${copyMsg} onEdit=${editMsg} onRegen=${regen} onMore=${more}/>
      <${RightPanel} bench=${status && status.bench} battle=${status && status.battle} board=${board}
        ready=${ready} onBench=${runBench} onBattle=${runBattle} open=${drawer === "panel"}/>
    </div>`;
}

mount(html`<${App}/>`, document.getElementById("app"));
