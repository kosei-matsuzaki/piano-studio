// editor.js — 楽譜エディタ。AI作曲で生成した曲のインポート・音符の自由な追加/編集・
// localStorage への保存を行う。保存した楽譜は「自由演奏」の参考楽譜からも参照できる。

// =================================================================
// 楽譜ライブラリ（localStorage。エディタ・自由演奏・AI作曲で共用）
// 曲データ: { id, title, tonicPc, bpm, treble:[note...], bass:[note...], updatedAt }
//   note = { midis:[MIDI...], beats, label? }（label=コード名表示） または { rest:true, beats }
// =================================================================
const ScoreLibrary = {
  KEY: "pianoScores.v1",
  _read() { try { return JSON.parse(localStorage.getItem(this.KEY) || "[]"); } catch (e) { return []; } },
  _write(list) { try { localStorage.setItem(this.KEY, JSON.stringify(list)); } catch (e) {} },
  list() { return this._read().sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)); },
  get(id) { return this._read().find((s) => s.id === id) || null; },
  save(song) {
    const list = this._read();
    if (!song.id) song.id = "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    song.updatedAt = Date.now();
    const i = list.findIndex((s) => s.id === song.id);
    if (i >= 0) list[i] = song; else list.push(song);
    this._write(list);
    return song;
  },
  remove(id) { this._write(this._read().filter((s) => s.id !== id)); },
};

// 曲データ → renderStaff 用スペック（上段・下段）。
// 両段を同じ拍数になるよう休符でパディングし、大譜表の小節割りがずれないようにする。
// パディングの休符は小節境界をまたがないよう「まず小節の残り→あとは1小節ずつ」で足す。
function songToStaffSpecs(song) {
  const tonic = song.tonicPc || 0;
  const spell = (m) => spellInKey(m, tonic);
  const toSpec = (n) => {
    if (n.rest) return { rest: true, beats: n.beats };
    const sorted = n.midis.slice().sort((a, b) => a - b);
    const spec = sorted.length > 1
      ? { chord: sorted.map(spell), beats: n.beats }
      : { name: spell(sorted[0]), beats: n.beats };
    if (n.label) spec.label = n.label;
    return spec;
  };
  const sum = (arr) => arr.reduce((s, n) => s + (n.beats || 0), 0);
  const tb = sum(song.treble || []), bb = sum(song.bass || []);
  const total = Math.max(4, Math.ceil(Math.max(tb, bb) / 4) * 4);
  const pad = (arr, cur) => {
    const out = arr.map(toSpec);
    let r = total - cur;
    const head = (4 - (cur % 4)) % 4;               // まず今の小節の残りを埋める
    if (r > 0 && head > 0) { const d = Math.min(head, r); out.push({ rest: true, beats: d }); r -= d; }
    while (r > 0.001) { const d = Math.min(4, r); out.push({ rest: true, beats: d }); r -= d; }
    return out;
  };
  return {
    treble: pad(song.treble || [], tb),
    bass: (song.bass && song.bass.length) ? pad(song.bass, bb) : null,
    totalBeats: total,
  };
}

// 曲データを大譜表（伴奏があれば2段）で描画して host を返す。
// host._marks / _bassMarks のインデックスは、曲データの音符インデックスと一致する
// （末尾パディングの休符ぶんだけ marks が長い）。
function renderSongStaff(song, fitWidth) {
  const specs = songToStaffSpecs(song);
  return renderStaff(specs.treble, {
    keySig: MAJOR_KEY[song.tonicPc || 0],
    timeSig: "4/4",
    barBeats: 4,
    measuresPerRow: 4,
    fitWidth: fitWidth || 600,
    bass: specs.bass,
  });
}

// 曲データを再生する共通ヘルパ（エディタ・自由演奏で共用）。
// 楽譜ハイライト（host._marks/_bassMarks）と鍵盤ヒント（ピッチクラス発光）を追従させる。
// 返り値: { stop() } のコントローラ。
function playSong(piano, song, host, opts) {
  opts = opts || {};
  const bpm = opts.bpm || song.bpm || 110;
  const beatMs = 60000 / bpm;
  const timers = [];
  const state = { mel: [], acc: [] };
  const refresh = () => piano.setHints([...state.mel, ...state.acc]);
  const marksOn = (marks, j) => (marks || []).forEach((m, k) => m && m.classList.toggle("on", k === j));
  // opts.scrollEl を渡すと、再生位置の段が見えるようスクロールを追従させる（エディタ用）
  const scrollEl = opts.scrollEl || null;
  let scrolledRow = -1;
  if (scrollEl) scrollEl.scrollTop = 0;
  const follow = (beatPos) => {
    if (!scrollEl || !host || !host._measureRow) return;
    const mi = Math.floor(beatPos / 4);
    const row = host._measureRow[mi];
    if (row == null || row === scrolledRow) return;
    scrolledRow = row;
    const anchor = (host._rowAnchor && host._rowAnchor[row]) || (host._measureMark && host._measureMark[mi]);
    if (!anchor) return;
    const delta = anchor.getBoundingClientRect().top - scrollEl.getBoundingClientRect().top - 12;
    if (Math.abs(delta) > 2) scrollEl.scrollBy({ top: delta, behavior: "smooth" });
  };
  const schedule = (notes, marks, vel, articulate) => {
    let cur = 0;
    notes.forEach((n, i) => {
      const pos = cur;
      timers.push(setTimeout(() => {
        follow(pos);
        marksOn(marks, i);
        if (articulate) state.mel = n.rest ? [] : n.midis; else state.acc = n.rest ? [] : n.midis;
        refresh();
        if (!n.rest) piano.playTones(n.midis.map(midiToName), n.beats * beatMs / 1000 * 0.93, vel);
      }, cur * beatMs));
      cur += n.beats;
    });
    return cur;
  };
  const t1 = schedule(song.treble || [], host && host._marks, 0.95, true);
  const t2 = schedule(song.bass || [], host && host._bassMarks, 0.7, false);
  // 再生カーソル（縦線）：一定テンポなので経過時間→拍位置をそのまま毎フレーム反映
  let cursorRaf = 0;
  const totalBeats = Math.max(t1, t2);
  if (host && host.setPlayCursor && totalBeats > 0) {
    const t0 = performance.now();
    const tick = () => {
      const b = (performance.now() - t0) / beatMs;
      if (b >= totalBeats) { host.setPlayCursor(-1); return; }
      host.setPlayCursor(Math.floor(b / 4), (b % 4) / 4);
      cursorRaf = requestAnimationFrame(tick);
    };
    cursorRaf = requestAnimationFrame(tick);
  }
  const finish = () => {
    state.mel = []; state.acc = [];
    piano.setHints([]);
    marksOn(host && host._marks, -1);
    marksOn(host && host._bassMarks, -1);
    if (cursorRaf) { cancelAnimationFrame(cursorRaf); cursorRaf = 0; }
    if (host && host.setPlayCursor) host.setPlayCursor(-1);
  };
  timers.push(setTimeout(() => { finish(); if (opts.onEnd) opts.onEnd(); }, Math.max(t1, t2) * beatMs + 150));
  return { stop() { timers.forEach(clearTimeout); finish(); } };
}

// =================================================================
// 楽譜エディタモード
// =================================================================
// 音価パレット（拍数と表示名）
const ED_DURS = [
  [4, "全"], [3, "付点2分"], [2, "2分"], [1.5, "付点4分"],
  [1, "4分"], [0.75, "付点8分"], [0.5, "8分"], [0.25, "16分"],
];
function edDurName(beats) {
  const hit = ED_DURS.find(([b]) => Math.abs(b - beats) < 0.001);
  return hit ? hit[1] : `${beats}拍`;
}

class EditorMode {
  constructor(piano) {
    this.piano = piano;
    this.song = null;
    this.sel = null;             // 選択中の音符 { track:"treble"|"bass", idx }
    this.track = "treble";       // 入力先トラック
    this.curBeats = 1;           // 入力する音価（拍）
    this.playCtl = null;
    this.prevActive = new Set(); // 直前の押鍵集合（新規押下の検出用）
    this.lastInsert = null;      // 直近に挿入した音符 { track, idx }（同時押しの和音まとめ用）
    this.lastInsertAt = 0;
    this.active = false;
    this.host = null;
    this.undoStack = [];         // 編集前スナップショットの履歴（Ctrl+Z）
    this.redoStack = [];

    this.songListEl = document.getElementById("ed-song-list");
    this.restoreEl = document.getElementById("ed-restore");
    this.restoreLabel = document.getElementById("ed-restore-label");
    this.lastDeleted = null;     // 直近に削除した曲（「元に戻す」用）
    this.titleInput = document.getElementById("ed-title");
    this.keySel = document.getElementById("ed-key");
    this.bpmSlider = document.getElementById("ed-bpm");
    this.bpmLabel = document.getElementById("ed-bpm-label");
    this.staffEl = document.getElementById("ed-staff");
    this.statusEl = document.getElementById("ed-status");
    this.inputChk = document.getElementById("ed-input");

    // キー（調号）の選択肢
    for (let pc = 0; pc < 12; pc++) {
      const o = document.createElement("option");
      o.value = pc; o.textContent = MAJOR_KEY[pc];
      this.keySel.appendChild(o);
    }

    // 音価パレット
    const dursEl = document.getElementById("ed-durs");
    this.durBtns = ED_DURS.map(([beats, label]) => {
      const b = document.createElement("button");
      b.className = "seg-btn" + (beats === this.curBeats ? " active" : "");
      b.dataset.beats = beats;
      b.textContent = label;
      b.title = `音価: ${label}音符`;
      b.addEventListener("click", () => this.setDur(beats));
      dursEl.appendChild(b);
      return b;
    });

    // 入力先トラック（メロディ／伴奏）
    this.trackBtns = [...document.querySelectorAll("#ed-track .seg-btn")];
    this.trackBtns.forEach((b) => b.addEventListener("click", () => this.setTrack(b.dataset.track)));

    // 曲の管理（一覧はクリックで開く。削除は各行の🗑 → deleteSong(id)）
    document.getElementById("ed-new").addEventListener("click", () => this.newSong());
    document.getElementById("ed-restore-btn").addEventListener("click", () => this.restoreDeleted());
    this.titleInput.addEventListener("input", () => {
      if (!this.song) return;
      this.song.title = this.titleInput.value.trim() || "無題の楽譜";
      this.save();
      const t = this.songListEl.querySelector(`.ed-song-item[data-id="${this.song.id}"] .ed-song-title`);
      if (t) t.textContent = this.song.title;
    });
    this.keySel.addEventListener("change", () => {
      if (!this.song) return;
      this.pushUndo();
      this.song.tonicPc = parseInt(this.keySel.value, 10) || 0;
      this.save(); this.render();
    });
    this.bpmSlider.addEventListener("input", () => {
      this.bpmLabel.textContent = this.bpmSlider.value;
      if (this.song) { this.song.bpm = parseInt(this.bpmSlider.value, 10) || 110; this.save(); }
    });

    // 編集ボタン
    this.undoBtn = document.getElementById("ed-undo");
    this.redoBtn = document.getElementById("ed-redo");
    this.undoBtn.addEventListener("click", () => this.undo());
    this.redoBtn.addEventListener("click", () => this.redo());
    document.getElementById("ed-add").addEventListener("click", () => this.addNote());
    document.getElementById("ed-rest").addEventListener("click", () => this.addRest());
    document.getElementById("ed-del").addEventListener("click", () => this.deleteSel());
    document.getElementById("ed-up").addEventListener("click", () => this.transposeSel(1));
    document.getElementById("ed-down").addEventListener("click", () => this.transposeSel(-1));
    document.getElementById("ed-octup").addEventListener("click", () => this.transposeSel(12));
    document.getElementById("ed-octdown").addEventListener("click", () => this.transposeSel(-12));

    // 再生
    document.getElementById("ed-play").addEventListener("click", () => this.play());
    document.getElementById("ed-stop").addEventListener("click", () => this.stopPlay());

    // 楽譜の余白クリックで選択解除（音符のハイライト枠クリックは select() 側で処理）
    this.staffEl.addEventListener("click", (e) => {
      if (e.target.classList && e.target.classList.contains("staff-note-hl")) return;
      if (this.sel) this.select(null);
    });

    // キーボードショートカット（エディタ表示中のみ）
    window.addEventListener("keydown", (e) => this.onKeydown(e));
  }

  // ウィンドウ幅の変化に合わせて段組みを引き直す（main.js から呼ばれる）
  onResize() {
    if (this.song) this.render();
  }

  // ---- 曲の管理 ----
  save() { if (this.song) ScoreLibrary.save(this.song); }

  // ---- アンドゥ/リドゥ ----
  // 編集操作の直前に snapshot を積む。タイトル・テンポの変更は対象外
  // （restore 時に現在の値を引き継ぎ、音符編集の取り消しで巻き戻らないようにする）。
  snapshot() {
    return { song: JSON.parse(JSON.stringify(this.song)), sel: this.sel ? { ...this.sel } : null };
  }

  pushUndo() {
    if (!this.song) return;
    this.undoStack.push(this.snapshot());
    if (this.undoStack.length > 100) this.undoStack.shift();
    this.redoStack.length = 0;
    this.updateUndoBtns();
  }

  restore(snap) {
    snap.song.title = this.song.title;
    snap.song.bpm = this.song.bpm;
    this.song = snap.song;
    this.sel = snap.sel;
    this.lastInsert = null;
    this.save();
    this.keySel.value = String(this.song.tonicPc || 0);
    this.render();
    this.updateUndoBtns();
  }

  undo() {
    if (!this.undoStack.length || !this.song) return;
    this.redoStack.push(this.snapshot());
    this.restore(this.undoStack.pop());
  }

  redo() {
    if (!this.redoStack.length || !this.song) return;
    this.undoStack.push(this.snapshot());
    this.restore(this.redoStack.pop());
  }

  updateUndoBtns() {
    if (this.undoBtn) this.undoBtn.disabled = !this.undoStack.length;
    if (this.redoBtn) this.redoBtn.disabled = !this.redoStack.length;
  }

  // 選択中・編集した音を耳で確認できるよう短く試聴する
  audition(n) {
    if (!n || n.rest || !this.piano.ready) return;
    this.piano.playTones(n.midis.map(midiToName), 0.35, 0.85);
  }

  // 楽譜ライブラリの一覧（左レール）を描き直す。開いている曲をハイライト。
  refreshSongList() {
    if (!this.songListEl) return;
    this.songListEl.innerHTML = "";
    const list = ScoreLibrary.list();
    if (!list.length) {
      const p = document.createElement("p");
      p.className = "ed-empty";
      p.textContent = "楽譜がまだありません。「＋ 新規」から作成できます。";
      this.songListEl.appendChild(p);
      return;
    }
    const sum = (arr) => (arr || []).reduce((x, n) => x + (n.beats || 0), 0);
    list.forEach((s) => {
      const item = document.createElement("div");
      item.className = "ed-song-item" + (this.song && s.id === this.song.id ? " active" : "");
      item.dataset.id = s.id;
      const main = document.createElement("div");
      main.className = "ed-song-main";
      const t = document.createElement("span");
      t.className = "ed-song-title";
      t.textContent = s.title || "無題の楽譜";
      const bars = Math.max(1, Math.ceil(Math.max(sum(s.treble), sum(s.bass)) / 4));
      const d = new Date(s.updatedAt || Date.now());
      const m = document.createElement("span");
      m.className = "ed-song-meta";
      m.textContent = `${bars}小節 ・ ${d.getMonth() + 1}/${d.getDate()} 更新`;
      main.appendChild(t); main.appendChild(m);
      const del = document.createElement("button");
      del.className = "ed-song-del";
      del.title = "この楽譜を削除";
      del.textContent = "🗑";
      del.addEventListener("click", (e) => { e.stopPropagation(); this.deleteSong(s.id); });
      item.appendChild(main); item.appendChild(del);
      const open = () => { if (!this.song || this.song.id !== s.id) this.loadSong(s.id); };
      item.addEventListener("click", open);
      // Tabフォーカス→Enterでも開ける（フォーカス中はホバーレールが開いたままになる）
      item.tabIndex = 0;
      item.setAttribute("role", "button");
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } });
      this.songListEl.appendChild(item);
    });
  }

  // 空のまま放置された「無題の楽譜」を自動で片付ける（一覧の散らかり防止）。
  // 削除したら true。
  cleanupIfEmpty(song) {
    if (!song) return false;
    const untitled = !song.title || song.title === "無題の楽譜";
    if (!untitled || (song.treble && song.treble.length) || (song.bass && song.bass.length)) return false;
    ScoreLibrary.remove(song.id);
    return true;
  }

  loadSong(id) {
    this.stopPlay();
    const s = ScoreLibrary.get(id);
    if (!s) return;
    if (this.song && this.song.id !== s.id) this.cleanupIfEmpty(this.song);
    s.treble = s.treble || []; s.bass = s.bass || [];
    this.song = s;
    this.sel = null; this.lastInsert = null;
    this.undoStack = []; this.redoStack = [];
    this.updateUndoBtns();
    this.titleInput.value = s.title || "";
    this.keySel.value = String(s.tonicPc || 0);
    const bpm = s.bpm || 110;
    this.bpmSlider.value = String(bpm); this.bpmLabel.textContent = String(bpm);
    this.refreshSongList();
    this.render();
  }

  newSong() {
    const song = ScoreLibrary.save({
      title: "無題の楽譜", tonicPc: this.piano.tonic || 0, bpm: 110, treble: [], bass: [],
    });
    this.loadSong(song.id);
  }

  deleteSong(id) {
    const s = ScoreLibrary.get(id);
    if (!s) return;
    if (!confirm(`「${s.title || "無題の楽譜"}」を削除しますか？`)) return;
    ScoreLibrary.remove(s.id);
    this.showRestore(s);
    if (this.song && this.song.id === s.id) {
      this.song = null;
      const first = ScoreLibrary.list()[0];
      if (first) this.loadSong(first.id); else this.newSong();
    } else {
      this.refreshSongList();
    }
  }

  // 削除直後の復元バー。しばらくすると自動で消える。
  showRestore(song) {
    this.lastDeleted = song;
    if (!this.restoreEl) return;
    this.restoreLabel.textContent = `「${song.title || "無題の楽譜"}」を削除しました`;
    this.restoreEl.hidden = false;
    clearTimeout(this._restoreT);
    this._restoreT = setTimeout(() => { this.restoreEl.hidden = true; this.lastDeleted = null; }, 15000);
  }

  restoreDeleted() {
    if (!this.lastDeleted) return;
    const s = ScoreLibrary.save(this.lastDeleted);
    this.lastDeleted = null;
    clearTimeout(this._restoreT);
    if (this.restoreEl) this.restoreEl.hidden = true;
    this.loadSong(s.id);
  }

  // ---- 描画 ----
  render() {
    this.stopPlay();
    this.staffEl.innerHTML = "";
    this.host = null;
    if (!this.song) { this.updateStatus(); return; }
    const host = renderSongStaff(this.song, this.staffEl.clientWidth || 700);
    host.classList.add("editable");
    this.host = host;
    this.staffEl.appendChild(host);
    // 音符クリックで選択。末尾パディングの休符（データ範囲外）はクリックで選択解除＝末尾入力。
    const wire = (marks, track) => {
      const count = (this.song[track] || []).length;
      (marks || []).forEach((m, i) => {
        if (!m) return;
        m.addEventListener("click", () => this.select(i < count ? { track, idx: i } : null));
      });
    };
    wire(host._marks, "treble");
    wire(host._bassMarks, "bass");
    this.applySel();
    this.scrollSelIntoView();
  }

  // 選択中の音符が #ed-staff のスクロール範囲外にあれば見える位置まで送る
  scrollSelIntoView() {
    if (!this.host || !this.sel) return;
    const marks = this.sel.track === "bass" ? this.host._bassMarks : this.host._marks;
    const m = marks && marks[this.sel.idx];
    if (!m) return;
    const box = this.staffEl.getBoundingClientRect();
    const r = m.getBoundingClientRect();
    const pad = 40;
    if (r.top < box.top + pad) this.staffEl.scrollTop += r.top - (box.top + pad);
    else if (r.bottom > box.bottom - pad) this.staffEl.scrollTop += r.bottom - (box.bottom - pad);
  }

  selNote() {
    if (!this.sel || !this.song) return null;
    return (this.song[this.sel.track] || [])[this.sel.idx] || null;
  }

  select(sel) {
    this.sel = sel;
    if (sel) {
      this.setTrack(sel.track);
      const n = this.selNote();
      if (n && ED_DURS.some(([b]) => Math.abs(b - n.beats) < 0.001)) this.curBeats = n.beats;
      this.audition(n);
    }
    this.applySel();
    this.scrollSelIntoView();
  }

  applySel() {
    const mark = (marks, track) => (marks || []).forEach((m, i) =>
      m && m.classList.toggle("sel", !!(this.sel && this.sel.track === track && this.sel.idx === i)));
    if (this.host) { mark(this.host._marks, "treble"); mark(this.host._bassMarks, "bass"); }
    this.durBtns.forEach((b) => b.classList.toggle("active", Math.abs(parseFloat(b.dataset.beats) - this.curBeats) < 0.001));
    this.trackBtns.forEach((b) => b.classList.toggle("active", b.dataset.track === this.track));
    this.updateStatus();
  }

  updateStatus() {
    if (!this.statusEl) return;
    if (!this.song) { this.statusEl.textContent = "—"; return; }
    const sum = (arr) => arr.reduce((s, n) => s + n.beats, 0);
    const bars = Math.max(1, Math.ceil(Math.max(sum(this.song.treble), sum(this.song.bass)) / 4));
    let txt = `${bars}小節 ・ メロディ ${this.song.treble.length}個 / 伴奏 ${this.song.bass.length}個`;
    const n = this.selNote();
    const trackName = this.sel && this.sel.track === "bass" ? "伴奏" : "メロディ";
    if (n) {
      txt += n.rest
        ? ` ｜ 選択: ${trackName} ${this.sel.idx + 1}番目の休符（${edDurName(n.beats)}）`
        : ` ｜ 選択: ${trackName} ${this.sel.idx + 1}番目 ${n.midis.map(midiToName).join("・")}（${edDurName(n.beats)}）`;
    } else {
      txt += " ｜ 選択なし（入力は末尾に追加されます）";
    }
    this.statusEl.textContent = txt;
  }

  // ---- 編集操作 ----
  setTrack(track) {
    if (track !== "treble" && track !== "bass") return;
    this.track = track;
    if (this.sel && this.sel.track !== track) this.sel = null;
    this.applySel();
  }

  // 音価を設定。音符を選択中ならその音符にも適用する（MuseScore等と同じ挙動）。
  setDur(beats) {
    const n = this.selNote();
    if (n) {
      this.pushUndo();
      this.curBeats = beats;
      n.beats = beats;
      this.save(); this.render();
    } else {
      this.curBeats = beats;
      this.applySel();
    }
  }

  // 選択の直後（未選択なら末尾）に音符を挿入し、挿入した音符を選択する
  insertAfterSel(note) {
    const arr = this.song[this.track];
    const at = (this.sel && this.sel.track === this.track) ? this.sel.idx + 1 : arr.length;
    arr.splice(at, 0, note);
    this.sel = { track: this.track, idx: at };
    this.save(); this.render();
    return at;
  }

  addNote() {
    if (!this.song) return;
    const arr = this.song[this.track];
    // 音高の初期値: 選択中の音 → 直近の音 → キーの主音（C4相当）
    const selN = (this.sel && this.sel.track === this.track) ? arr[this.sel.idx] : null;
    const ref = (selN && !selN.rest) ? selN : [...arr].reverse().find((n) => !n.rest);
    const midis = ref ? ref.midis.slice() : [60 + (this.song.tonicPc || 0)];
    this.lastInsert = null;
    this.pushUndo();
    this.insertAfterSel({ midis, beats: this.curBeats });
    this.audition(this.selNote());
  }

  addRest() {
    if (!this.song) return;
    this.lastInsert = null;
    this.pushUndo();
    this.insertAfterSel({ rest: true, beats: this.curBeats });
  }

  deleteSel() {
    const n = this.selNote();
    if (!n) return;
    this.pushUndo();
    const { track, idx } = this.sel;
    this.song[track].splice(idx, 1);
    const len = this.song[track].length;
    this.sel = len ? { track, idx: Math.min(idx, len - 1) } : null;
    this.lastInsert = null;
    this.save(); this.render();
  }

  transposeSel(d) {
    const n = this.selNote();
    if (!n || n.rest) return;
    this.pushUndo();
    n.midis = [...new Set(n.midis.map((m) => Math.max(21, Math.min(108, m + d))))].sort((a, b) => a - b);
    this.audition(n);
    this.save(); this.render();
  }

  move(d) {
    if (!this.song) return;
    const arr = this.song[this.track];
    if (!arr.length) return;
    let idx = (this.sel && this.sel.track === this.track)
      ? this.sel.idx + d
      : (d > 0 ? 0 : arr.length - 1);
    idx = Math.max(0, Math.min(arr.length - 1, idx));
    this.select({ track: this.track, idx });
  }

  onKeydown(e) {
    if (!this.active) return;
    const tag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag)) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (e.key === "z" || e.key === "Z")) { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); }
    else if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); this.redo(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); this.move(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); this.move(1); }
    // ↑↓ = 選択音符を半音（Shift でオクターブ）移調。未選択時は preventDefault せず、
    // main.js 側のオクターブ変更に委ねる。
    else if (e.key === "ArrowUp" && this.selNote() && !this.selNote().rest) { e.preventDefault(); this.transposeSel(e.shiftKey ? 12 : 1); }
    else if (e.key === "ArrowDown" && this.selNote() && !this.selNote().rest) { e.preventDefault(); this.transposeSel(e.shiftKey ? -12 : -1); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); this.deleteSel(); }
    // Space = 再生/停止（ボタンにフォーカスがあるときはボタンのクリックを優先）
    else if (e.key === " " && tag !== "button") { e.preventDefault(); this.playCtl ? this.stopPlay() : this.play(); }
    // Esc = 再生停止 → 選択解除
    else if (e.key === "Escape") { e.preventDefault(); if (this.playCtl) this.stopPlay(); else this.select(null); }
  }

  // ---- 鍵盤からの音符入力 ----
  // 下部ピアノ（PCキー/マウス）の押鍵が届く。新規押下を選択位置の直後に挿入する。
  // 直前の挿入から間を置かず重ねて押した鍵は、同じ音符に積んで和音にする。
  update(notes) {
    const newly = notes.filter((m) => !this.prevActive.has(m));
    const wasHeld = this.prevActive.size > 0;
    this.prevActive = new Set(notes);
    if (!notes.length) this.lastInsert = null;
    if (!newly.length || !this.song) return;
    if (!this.inputChk || !this.inputChk.checked) return;
    if (this.playCtl) return;  // 再生中は挿入しない（弾いても曲を壊さない）

    const now = performance.now();
    const arr = this.song[this.track];
    const li = this.lastInsert;
    // 和音判定: 前の鍵を押したまま、ごく短い間隔で次の鍵 → 直前の音符に積む
    if (wasHeld && li && li.track === this.track && arr[li.idx] && !arr[li.idx].rest
        && now - this.lastInsertAt < 200) {
      // 和音への積み増しはアンドゥを積まない（Ctrl+Z 一回で和音全体の挿入が取り消せる）
      const note = arr[li.idx];
      note.midis = [...new Set([...note.midis, ...newly])].sort((a, b) => a - b);
      this.lastInsertAt = now;
      this.save(); this.render();
      return;
    }
    this.pushUndo();
    const at = this.insertAfterSel({ midis: newly.slice().sort((a, b) => a - b), beats: this.curBeats });
    this.lastInsert = { track: this.track, idx: at };
    this.lastInsertAt = now;
  }

  // ---- 再生 ----
  async play() {
    if (!this.song) return;
    await this.piano.ensureAudio();
    this.stopPlay();
    const bpm = parseInt(this.bpmSlider.value, 10) || 110;
    this.playCtl = playSong(this.piano, this.song, this.host, {
      bpm,
      scrollEl: this.staffEl,
      onEnd: () => { this.playCtl = null; },
    });
  }

  stopPlay() {
    if (this.playCtl) { this.playCtl.stop(); this.playCtl = null; }
  }

  enter() {
    this.active = true;
    this.prevActive = new Set();
    if (!this.song) {
      const first = ScoreLibrary.list()[0];
      if (first) this.loadSong(first.id); else this.newSong();
    } else {
      this.refreshSongList();
      this.render();
    }
  }

  leave() {
    this.active = false;
    this.stopPlay();
    this.piano.setHints([]);
    // 空のまま離れた「無題の楽譜」は片付ける（自由演奏の参考楽譜一覧も汚さない）
    if (this.cleanupIfEmpty(this.song)) this.song = null;
  }
}
