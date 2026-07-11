// main.js — 全体の組み立て。鍵盤・モード・キーボード入力・タブ切り替えをつなぐ。

const piano = new Piano(document.getElementById("piano"));

// 各モードを生成
const modes = {
  free: new FreeMode(piano),
  learn: new LearnMode(piano),
  compose: new ComposeMode(piano),
  editor: new EditorMode(piano),
};

let currentMode = "free";

// 鍵盤の状態が変わるたびに、今アクティブなモードへ通知
piano.onChange((notes) => {
  const m = modes[currentMode];
  if (m && typeof m.update === "function") m.update(notes);
});

// ---- タブ切り替え ----
const tabs = document.querySelectorAll(".tab");
const panels = {
  free: document.getElementById("panel-free"),
  learn: document.getElementById("panel-learn"),
  compose: document.getElementById("panel-compose"),
  editor: document.getElementById("panel-editor"),
};

function switchMode(mode) {
  if (mode === currentMode) return;
  // 退出処理
  const prev = modes[currentMode];
  if (prev && typeof prev.leave === "function") prev.leave();

  currentMode = mode;
  document.body.dataset.mode = mode;   // CSSがモード別レイアウト（サイドレールの余白等）に使う
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  Object.entries(panels).forEach(([k, el]) => el.classList.toggle("active", k === mode));

  const next = modes[mode];
  if (next && typeof next.enter === "function") next.enter();
}

tabs.forEach((t) => t.addEventListener("click", () => switchMode(t.dataset.mode)));
document.body.dataset.mode = currentMode;

// ---- ウィンドウリサイズ ----
// 画面左端ドッキングのサイドレール用に、ヘッダー・下部ドックの実高さをCSS変数へ反映
const appBarEl = document.querySelector(".app-bar");
const pianoWrapEl = document.querySelector(".piano-wrap");
function updateRailBounds() {
  const st = document.documentElement.style;
  if (appBarEl) st.setProperty("--rail-top", appBarEl.offsetHeight + "px");
  if (pianoWrapEl) st.setProperty("--rail-bottom", pianoWrapEl.offsetHeight + "px");
}
updateRailBounds();
window.addEventListener("load", updateRailBounds);  // フォント読込等で高さが変わった後にも追従

// 画面幅は固定ではなくウィンドウに追従するため、幅に合わせて描画し直す必要がある
// モード（楽譜の段組みなど）へデバウンスして通知する。
let resizeTimer = null;
window.addEventListener("resize", () => {
  updateRailBounds();
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const m = modes[currentMode];
    if (m && typeof m.onResize === "function") m.onResize();
  }, 200);
});

// 他モードから「楽譜エディタで開く」ためのAPI（AI作曲 → エディタへ送る で使用）
window.appOpenInEditor = (songId) => {
  switchMode("editor");
  if (songId) modes.editor.loadSong(songId);
};

// ---- 音色(楽器)選択 ----
const instSel = document.getElementById("instrument");
for (const [key, def] of Object.entries(INSTRUMENTS)) {
  const o = document.createElement("option");
  o.value = key; o.textContent = def.label;
  instSel.appendChild(o);
}
instSel.value = piano.instrument;
instSel.addEventListener("change", () => piano.setInstrument(instSel.value));

// ---- キー(調) = Aキー(ド)の音の選択 ----
const tonicSel = document.getElementById("tonic");
const TONIC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
TONIC_NAMES.forEach((n, pc) => {
  const o = document.createElement("option");
  o.value = pc; o.textContent = n;
  tonicSel.appendChild(o);
});
tonicSel.value = piano.tonic;
tonicSel.addEventListener("change", () => {
  piano.setTonic(parseInt(tonicSel.value, 10));
  const m = modes[currentMode];
  if (m && typeof m.onTranspose === "function") m.onTranspose();
});

// ---- オクターブ操作 ----
document.getElementById("oct-down").addEventListener("click", () => piano.shiftOctave(-1));
document.getElementById("oct-up").addEventListener("click", () => piano.shiftOctave(1));

// ---- PCキーボード入力 ----
// 押しっぱなしのリピートを無視するため、押下中のキーを管理
const heldKeys = new Set();

window.addEventListener("keydown", async (e) => {
  if (e.repeat) return;
  // モード側（楽譜エディタ等）が処理済みのキーは無視
  if (e.defaultPrevented) return;
  // フォーム操作中（select等）はピアノ入力を無効化
  if (["select", "input", "textarea"].includes(document.activeElement.tagName.toLowerCase())) return;

  // 矢印上下でオクターブ変更
  if (e.key === "ArrowUp") { e.preventDefault(); piano.shiftOctave(1); return; }
  if (e.key === "ArrowDown") { e.preventDefault(); piano.shiftOctave(-1); return; }

  // Ctrl/Cmd/Alt 付きのショートカット（コピー・アンドゥ等）で発音しない
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  const key = e.key.toLowerCase();
  const off = piano.offsetForKey(key);
  if (off === undefined) return;
  e.preventDefault();
  await piano.ensureAudio();
  if (heldKeys.has(key)) return;
  heldKeys.add(key);
  piano.noteOn(off);
});

window.addEventListener("keyup", (e) => {
  const key = e.key.toLowerCase();
  const off = piano.offsetForKey(key);
  if (off === undefined) return;
  heldKeys.delete(key);
  piano.noteOff(off);
});

// マウス操作でも音声を有効化
document.body.addEventListener("mousedown", () => piano.ensureAudio(), { once: true });

// 初期表示
modes.free.update([]);
