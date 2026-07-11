// modes.js — 各モードのロジック
// 「自由演奏」と、スライド式「コード学習」。カリキュラム本体は curriculum.js。
// 鍵盤の解説・練習は画面下部のピアノで行い、スライドの key に応じて自動で調を切り替える。

const ROOT_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// 異名同音（♯→♭）。臨時記号が少なくなる方を選んで読みやすく綴る。
const SHARP_TO_FLAT = { "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb" };
function accCount(names) {
  return names.reduce((s, n) => s + ((n.match(/[#b]/g) || []).length), 0);
}

// =================================================================
// 自由演奏モード
// =================================================================
class FreeMode {
  constructor(piano) {
    this.piano = piano;
    this.notesEl = document.getElementById("free-notes");
    this.chordEl = document.getElementById("free-chord");

    // 参考楽譜（AI作曲・楽譜エディタで保存した曲を表示して、見ながら演奏する）
    this.songSel = document.getElementById("fs-song");
    this.staffEl = document.getElementById("fs-staff");
    this.songData = null;
    this.host = null;
    this.ctl = null;           // 再生コントローラ（playSong の返り値）
    if (this.songSel) {
      this.songSel.addEventListener("change", () => this.showScore());
      document.getElementById("fs-play").addEventListener("click", () => this.playScore());
      document.getElementById("fs-stop").addEventListener("click", () => this.stopScore());
      // 初期表示（freeモードは起動時に enter() が呼ばれないため、ここで一覧を反映）
      setTimeout(() => this.refreshScoreList(), 0);
    }
  }
  update(notes) {
    this.notesEl.textContent = notes.length ? notes.map(midiToName).join("  ") : "—";
    const chord = detectChord(notes);
    this.chordEl.textContent = chord || (notes.length >= 3 ? "(該当なし)" : "—");
  }

  // ライブラリの一覧をセレクタに反映（選択は維持）。ScoreLibrary は editor.js 由来。
  refreshScoreList() {
    if (!this.songSel || typeof ScoreLibrary === "undefined") return;
    const cur = this.songSel.value;
    this.songSel.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = ""; ph.textContent = "（楽譜を選ぶ）";
    this.songSel.appendChild(ph);
    ScoreLibrary.list().forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id; o.textContent = s.title || "無題の楽譜";
      this.songSel.appendChild(o);
    });
    this.songSel.value = cur && ScoreLibrary.get(cur) ? cur : "";
    this.showScore();
  }

  showScore() {
    this.stopScore();
    const id = this.songSel ? this.songSel.value : "";
    this.songData = id ? ScoreLibrary.get(id) : null;
    if (!this.staffEl) return;
    this.staffEl.innerHTML = "";
    this.host = null;
    if (!this.songData) { this.staffEl.hidden = true; return; }
    this.staffEl.hidden = false;
    this.host = renderSongStaff(this.songData, this.staffEl.clientWidth || 700);
    this.staffEl.appendChild(this.host);
  }

  async playScore() {
    if (!this.songData) return;
    await this.piano.ensureAudio();
    this.stopScore();
    this.ctl = playSong(this.piano, this.songData, this.host, {
      bpm: this.songData.bpm,
      onEnd: () => { this.ctl = null; },
    });
  }

  stopScore() {
    if (this.ctl) { this.ctl.stop(); this.ctl = null; }
  }

  // ウィンドウ幅の変化で参考楽譜の段組みを引き直す（再生中はハイライト参照が壊れるため見送る）
  onResize() {
    if (this.songData && !this.ctl) this.showScore();
  }

  enter() { this.refreshScoreList(); }
  leave() { this.stopScore(); }
}

// =================================================================
// コード学習モード（スライド式）
// =================================================================
const SAVE_KEY = "pianoCurriculumDone";

class LearnMode {
  constructor(piano) {
    this.piano = piano;
    this.listEl = document.getElementById("lesson-list");
    this.topicEl = document.getElementById("slide-topic");
    this.titleEl = document.getElementById("lesson-title");
    this.counterEl = document.getElementById("slide-counter");
    this.bodyEl = document.getElementById("slide-body");
    this.actionsEl = document.getElementById("slide-actions");
    this.dotsEl = document.getElementById("slide-dots");
    this.prevBtn = document.getElementById("slide-prev");
    this.nextBtn = document.getElementById("slide-next");
    this.tonicSel = document.getElementById("tonic");

    this.topicIndex = 0;
    this.slideIndex = 0;
    this.completed = this.load();
    this.timers = [];

    // 練習状態
    this.pSeq = null;     // 配列（chords / notes(offset) / voicings(offset配列)）
    this.pKind = null;    // "chords" | "notes" | "voicings"
    this.pIndex = 0;
    this.locked = false;
    this.prevActive = new Set();
    this.refs = {};
    this.animTimers = [];
    this.exTimers = [];        // 応用例（メロディ）再生用タイマー
    this.animHost = null;
    this.replayBtn = null;
    this.exampleBtn = null;
    this._animBusy = false;
    this._exBusy = false;
    this._winIdx = -1;   // 窓を計算したPartのインデックス（キャッシュ用）

    this.prevBtn.addEventListener("click", () => this.go(-1));
    this.nextBtn.addEventListener("click", () => this.go(1));
    this.renderList();
  }

  load() { try { return new Set(JSON.parse(localStorage.getItem(SAVE_KEY) || "[]")); } catch (e) { return new Set(); } }
  save() { try { localStorage.setItem(SAVE_KEY, JSON.stringify([...this.completed])); } catch (e) {} }
  clearTimers() { this.timers.forEach((t) => clearTimeout(t)); this.timers = []; }

  get topic() { return CURRICULUM[this.topicIndex]; }
  get slide() { return this.topic.slides[this.slideIndex]; }

  transposeRoot(root) { return NOTE_NAMES[(NOTE_NAMES.indexOf(root) + this.piano.tonic) % 12]; }
  dispName(root, type) { return makeChordName(this.transposeRoot(root), type); }
  chordMidi(root, type) { return chordNotes(this.piano.baseMidi() + NOTE_NAMES.indexOf(root), type); }

  // 下部ピアノの窓（オクターブ＋左端音）を設定し、セレクトも同期
  setWindowUI(octave, tonic) {
    this.piano.setWindow(octave, ((tonic % 12) + 12) % 12);
    if (this.tonicSel) this.tonicSel.value = String(this.piano.tonic);
  }

  // このPartの全音符（練習以外のstaff/keys）が鍵盤に収まる窓を計算
  computePartWindow(topic) {
    const ms = [];
    topic.slides.forEach((s) => {
      if (s.practice) return;
      if (s.staff) s.staff.forEach((e) => normalizeEntry(e).names.forEach((nm) => ms.push(noteNameToMidi(nm))));
      if (s.keys) s.keys.forEach((nm) => ms.push(noteNameToMidi(nm)));
    });
    if (!ms.length) return { octave: 4, tonic: 0 };
    const minM = Math.min(...ms), maxM = Math.max(...ms);
    let base = maxM - minM > DISPLAY_MAX ? Math.max(minM, maxM - DISPLAY_MAX) : minM;
    base = Math.max(24, Math.min(96, base));
    return { octave: Math.floor(base / 12) - 1, tonic: ((base % 12) + 12) % 12 };
  }

  // Partの窓を適用（Partが変わったときだけ計算。同Part内のスライドでは動かない）
  applyPartWindow() {
    if (this._winIdx !== this.topicIndex) { this._win = this.computePartWindow(this.topic); this._winIdx = this.topicIndex; }
    this.setWindowUI(this._win.octave, this._win.tonic);
  }

  // ---- トピック一覧 ----
  renderList() {
    this.listEl.innerHTML = "";
    let lastChapter = null;
    CURRICULUM.forEach((t, i) => {
      if (t.chapter && t.chapter !== lastChapter) {
        lastChapter = t.chapter;
        const h = document.createElement("div");
        h.className = "lesson-chapter";
        h.textContent = t.chapter;
        this.listEl.appendChild(h);
      }
      const done = this.completed.has(t.title);
      const btn = document.createElement("button");
      btn.className = "lesson-item" + (i === this.topicIndex ? " active" : "") + (done ? " done" : "");
      // ステータス丸（完了で✓）＋タイトル
      btn.innerHTML = `<span class="li-mark"></span><span class="li-title"></span>`;
      btn.querySelector(".li-title").textContent = t.title;
      btn.addEventListener("click", () => this.selectTopic(i));
      this.listEl.appendChild(btn);
    });
    this.updateProgress();
  }

  // 全体進捗（完了トピック数 / 全トピック数）をサイドバー上部に反映
  updateProgress() {
    const fill = document.getElementById("learn-progress-fill");
    const text = document.getElementById("learn-progress-text");
    if (!fill || !text) return;
    const total = CURRICULUM.length;
    const done = CURRICULUM.reduce((n, t) => n + (this.completed.has(t.title) ? 1 : 0), 0);
    fill.style.width = total ? `${(done / total) * 100}%` : "0%";
    text.textContent = `${done} / ${total}`;
  }

  selectTopic(i) { this.topicIndex = i; this.slideIndex = 0; this.renderList(); this.showSlide(); }

  go(dir) {
    const len = this.topic.slides.length;
    const s = this.slideIndex + dir;
    if (s >= len) { if (this.topicIndex + 1 < CURRICULUM.length) this.selectTopic(this.topicIndex + 1); return; }
    if (s < 0) {
      if (this.topicIndex > 0) { this.topicIndex--; this.slideIndex = CURRICULUM[this.topicIndex].slides.length - 1; this.renderList(); this.showSlide(); }
      return;
    }
    this.slideIndex = s; this.showSlide();
  }

  // ---- スライド描画 ----
  showSlide() {
    this.clearTimers();
    this.pSeq = null; this.pKind = null; this.locked = false; this.prevActive = new Set();
    this.pStaffWrap = null;
    this.rhythmRows = null; this.rhythmBtn = null; this._rhythmBusy = false;
    this.exampleBtn = null;
    if (this.actionsEl) this.actionsEl.innerHTML = "";
    this.clearAnim();

    const slide = this.slide;
    const len = this.topic.slides.length;
    // 練習：その調・標準オクターブへ。 解説：Part内の全音符が収まる窓に固定（スライド間で動かさない）
    if (slide.practice) this.setWindowUI(4, NOTE_NAMES.indexOf(slide.key || "C"));
    else this.applyPartWindow();

    this.topicEl.textContent = this.topic.title;
    this.titleEl.textContent = slide.heading;
    this.counterEl.textContent = `${this.slideIndex + 1} / ${len}`;

    this.dotsEl.innerHTML = "";
    for (let i = 0; i < len; i++) {
      const d = document.createElement("span");
      d.className = "slide-dot" + (i === this.slideIndex ? " active" : "");
      this.dotsEl.appendChild(d);
    }

    this.bodyEl.innerHTML = "";
    if (slide.body) {
      const ul = document.createElement("ul");
      ul.className = "slide-bullets";
      slide.body.forEach((b) => { const li = document.createElement("li"); li.textContent = b; ul.appendChild(li); });
      this.bodyEl.appendChild(ul);
    }
    this.animHost = null;
    if (slide.staff) {
      const w = document.createElement("div");
      w.className = "visual-wrap";
      this.animHost = renderStaff(slide.staff, slide.staffOpts || {});
      w.appendChild(this.animHost);
      this.bodyEl.appendChild(w);
    }
    this.replayBtn = null;
    if (slide.practice) {
      this.buildPractice(slide.practice);
    } else if (slide.rhythm) {
      // リズム：拍のグリッドを表示し、指定拍位置でクリック音を鳴らす。
      this.buildRhythm(slide.rhythm);
    } else {
      // 解説スライド：内容に応じて「概念デモ（鍵盤アニメ）」「応用例メロディ」を過不足なく出す。
      //  - 違い・流れ・音階など2音以上 → 概念デモを自動再生。単音／単一和音／音材なし → 楽譜のみ。
      //  - 応用例は手書き example があるスライドだけ。
      const willAnim = this.shouldAnimate(slide);
      const hasExample = !!this.exampleSpec(slide);
      const bar = document.createElement("div");
      bar.className = "anim-btns";
      if (willAnim) {
        const b1 = document.createElement("button");
        b1.className = "btn ghost anim-replay";
        b1.textContent = "▶ 解説を再生";
        b1.addEventListener("click", () => this.startAnim());
        bar.appendChild(b1);
        this.replayBtn = b1;
      }
      if (hasExample) {
        const b2 = document.createElement("button");
        b2.className = "btn ghost example-btn";
        b2.textContent = "🎵 応用例を聴く";
        b2.addEventListener("click", () => this.playExample());
        bar.appendChild(b2);
        this.exampleBtn = b2;
      }
      if (bar.children.length) this.actionsEl.appendChild(bar);

      let dur = 0;
      if (willAnim) dur = this.startAnim() || 0;           // 概念デモ（1回）
      else this.piano.setHints([]);                        // 楽譜のみ：鍵盤ハイライトは出さない
      if (hasExample) this.exTimers.push(setTimeout(() => this.playExample(), willAnim ? dur + 400 : 350));
    }

    const lastTopic = this.topicIndex + 1 >= CURRICULUM.length;
    const lastSlide = this.slideIndex + 1 >= len;
    this.nextBtn.textContent = lastSlide ? (lastTopic ? "完了" : "次のパートへ →") : "次へ →";
    this.nextBtn.disabled = lastSlide && lastTopic;
    this.prevBtn.disabled = this.slideIndex === 0 && this.topicIndex === 0;

    if (lastSlide) { this.completed.add(this.topic.title); this.save(); this.renderList(); }
  }

  // ---- 楽譜＋鍵盤の自動アニメーション（練習でないスライド）----
  clearAnim() {
    (this.animTimers || []).forEach((t) => clearTimeout(t));
    (this.exTimers || []).forEach((t) => clearTimeout(t));
    this.animTimers = [];
    this.exTimers = [];
    this._animBusy = false;
    this._exBusy = false;
  }

  // スライドの内容を「グループ（音符/和音）の配列」に。各要素は鍵盤オフセットの配列。
  animGroups() {
    const s = this.slide;
    const base = this.piano.baseMidi();
    const tonic = this.piano.tonic;
    const keyOffset = (nm) => {
      const midi = noteNameToMidi(nm);
      const raw = midi - base;
      if (raw >= 0 && raw <= DISPLAY_MAX) return raw;               // 実位置（オクターブ保持）
      return ((pitchClass(midi) - tonic) % 12 + 12) % 12;           // 範囲外はpcで最寄り
    };
    if (s.staff) return s.staff.map((e) => normalizeEntry(e).names.map(keyOffset));
    if (s.voicing) return [s.voicing.slice()];
    if (s.keys) return s.keys.map((nm) => [keyOffset(nm)]);
    return [];
  }

  startAnim() {
    this.clearAnim();
    const groups = this.animGroups();
    this.piano.setHints([]);
    if (!groups.length) { this._animBusy = false; this._updateReplayBtn(); return 0; }
    this.piano.ensureAudio();
    const marks = (this.animHost && this.animHost._marks) || [];
    const base = this.piano.baseMidi();
    // NOTE_STEP:単音の間隔, CHORD_STEP:和音の間隔（和音は最初から同時に鳴らす）
    const NOTE_STEP = 470, CHORD_STEP = 1050;

    this._animBusy = true; this._updateReplayBtn();
    let t = 120;                                   // 音声開始のわずかな余裕
    groups.forEach((g, gi) => {
      const midis = g.map((off) => base + off);
      this.animTimers.push(setTimeout(() => {
        this.piano.setHints([]);
        marks.forEach((m, j) => { if (m) m.classList.toggle("on", j === gi); });
        if (midis.length) {
          this.piano.setHintsMidi(midis);                                   // 和音は全鍵盤を同時に光らせる
          this.piano.playTones(midis.map(midiToName), g.length > 1 ? 0.9 : 0.55);  // 同時発音
        }
      }, t));
      t += g.length > 1 ? CHORD_STEP : NOTE_STEP;
    });
    // 一度終わったら止める（ループしない）
    this.animTimers.push(setTimeout(() => {
      this.piano.setHints([]);
      marks.forEach((m) => { if (m) m.classList.remove("on"); });
      this._animBusy = false; this._updateReplayBtn();
    }, t + 300));
    return t + 300;
  }

  // ---- 応用例（その概念を使った短いメロディ＋伴奏）----
  _updateExampleBtn() {
    if (!this.exampleBtn) return;
    this.exampleBtn.textContent = this._exBusy ? "🎵 再生中…" : "🎵 応用例を聴く";
    this.exampleBtn.disabled = this._exBusy;
  }

  // 応用例のフレーズspecを返す。手書き example があるスライドだけ（無ければ null＝応用例なし）。
  exampleSpec(slide) {
    return slide.example ? this.normExample(slide.example) : null;
  }

  // 概念デモ（鍵盤アニメ＋発音）を自動再生すべきか。
  //  - 違い・流れ・音階など「2音以上の並び」→ 再生。
  //  - 単一和音でも、7th／テンションなど4音以上のリッチな響きは聴かせる（「◯◯の響き」系）。
  //  - 単音・単一トライアド・音材なしは楽譜のみ。anim:true/false で明示上書きも可能。
  shouldAnimate(slide) {
    if (slide.anim === false) return false;
    if (slide.anim === true) return true;
    const g = this.animGroups();
    if (g.length >= 2) return true;
    if (g.length === 1 && g[0].length >= 4) return true;
    return false;
  }

  // 手書き example を正規化。melody要素: [off,beats] / {off,beats} / {rest:true,beats} / "r<beats>"
  normExample(ex) {
    const melody = (ex.melody || []).map((m) => {
      if (Array.isArray(m)) return { off: m[0], beats: m[1] == null ? 1 : m[1] };
      if (typeof m === "string") return { rest: true, beats: parseFloat(m.slice(1)) || 1 };
      if (m.rest) return { rest: true, beats: m.beats || 1 };
      return { off: m.off, beats: m.beats == null ? 1 : m.beats };
    });
    const chords = (ex.chords || []).map((c) =>
      Array.isArray(c) ? { offs: c, beats: ex.chordBeats || 4 } : { offs: c.offs, beats: c.beats || 4 });
    if (!melody.length && !chords.length) return null;
    return { tempo: ex.tempo || 100, melody, chords, label: ex.label || null };
  }

  playExample() {
    this.clearAnim();
    const spec = this.exampleSpec(this.slide);
    if (!spec) { this._exBusy = false; this._updateExampleBtn(); return; }
    this.piano.ensureAudio();
    const base = this.piano.baseMidi();
    const beatMs = 60000 / (spec.tempo || 100);
    this._exBusy = true; this._updateExampleBtn();

    // 伴奏（和音）：1オクターブ下で各コードを順に持続。低くなりすぎる場合は同オクターブ。
    let cb = 0;
    (spec.chords || []).forEach((c) => {
      const startMs = cb * beatMs;
      const durSec = Math.max(0.2, c.beats * beatMs / 1000 * 0.98);
      let midis = c.offs.map((o) => base + o - 12);
      if (Math.min(...midis) < 33) midis = c.offs.map((o) => base + o);
      this.exTimers.push(setTimeout(() => this.piano.playTones(midis.map(midiToName), durSec), startMs));
      cb += c.beats;
    });

    // メロディ：拍に合わせて発音＋鍵盤ハイライト
    let mb = 0;
    spec.melody.forEach((n) => {
      const startMs = mb * beatMs;
      if (!n.rest) {
        const midi = base + n.off;
        const durSec = Math.max(0.12, n.beats * beatMs / 1000 * 0.92);
        this.exTimers.push(setTimeout(() => {
          this.piano.setHintsMidi([midi]);
          this.piano.playTones([midiToName(midi)], durSec);
        }, startMs));
      }
      mb += n.beats;
    });

    const totalMs = Math.max(mb, cb) * beatMs;
    this.exTimers.push(setTimeout(() => {
      this.piano.setHints([]);
      this._exBusy = false; this._updateExampleBtn();
    }, totalMs + 260));
  }

  _updateReplayBtn() {
    if (!this.replayBtn) return;
    this.replayBtn.textContent = this._animBusy ? "▶ 再生中…" : "▶ もう一度再生";
    this.replayBtn.disabled = this._animBusy;
  }

  // ---- リズム（拍のグリッド＋クリック再生）----
  // rhythm: { tempo, beats(1小節の拍数), bars(小節数), grid(1拍の分割数), loop,
  //           layers:[{ name, pitch, accentPitch, hits:[拍位置...], accents:[拍位置...] }] }
  buildRhythm(r) {
    this.stopRhythm();
    const grid = r.grid || 4;
    const totalBeats = (r.beats || 4) * (r.bars || 1);
    const cols = Math.round(totalBeats * grid);

    const wrap = document.createElement("div");
    wrap.className = "rhythm-wrap";
    this.rhythmRows = [];
    (r.layers || []).forEach((L) => {
      const hitCols = new Set((L.hits || []).map((h) => Math.round(h * grid)));
      const accCols = new Set((L.accents || []).map((h) => Math.round(h * grid)));
      const row = document.createElement("div");
      row.className = "rhythm-row";
      const nameEl = document.createElement("span");
      nameEl.className = "rhythm-name";
      nameEl.textContent = L.name || "";
      row.appendChild(nameEl);
      const cells = [];
      const gridEl = document.createElement("div");
      gridEl.className = "rhythm-grid";
      gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      for (let ci = 0; ci < cols; ci++) {
        const cell = document.createElement("div");
        cell.className = "rhythm-cell";
        if (ci % grid === 0) cell.classList.add("beat-start");
        if (ci % (grid * (r.beats || 4)) === 0 && ci > 0) cell.classList.add("bar-start");
        if (hitCols.has(ci)) cell.classList.add("hit");
        if (accCols.has(ci)) cell.classList.add("accent");
        gridEl.appendChild(cell);
        cells.push(cell);
      }
      row.appendChild(gridEl);
      wrap.appendChild(row);
      this.rhythmRows.push(cells);
    });
    this.bodyEl.appendChild(wrap);

    const btn = document.createElement("button");
    btn.className = "btn ghost anim-replay";
    btn.textContent = "▶ リズムを再生";
    btn.addEventListener("click", () => { if (this._rhythmBusy) this.stopRhythm(); else this.playRhythm(r); });
    this.actionsEl.appendChild(btn);
    this.rhythmBtn = btn;
    this.playRhythm(r);
  }

  _updateRhythmBtn() {
    if (!this.rhythmBtn) return;
    this.rhythmBtn.textContent = this._rhythmBusy ? "■ 停止" : "▶ リズムを再生";
  }

  stopRhythm() {
    (this.animTimers || []).forEach((t) => clearTimeout(t));
    this.animTimers = [];
    this._rhythmBusy = false;
    (this.rhythmRows || []).forEach((cells) => cells.forEach((c) => c.classList.remove("cur")));
    this._updateRhythmBtn();
  }

  playRhythm(r) {
    this.stopRhythm();
    this.piano.ensureAudio();
    const grid = r.grid || 4;
    const beatMs = 60000 / (r.tempo || 90);
    const cellMs = beatMs / grid;
    const totalBeats = (r.beats || 4) * (r.bars || 1);
    const cols = Math.round(totalBeats * grid);
    const loops = r.loop || 2;
    const layers = (r.layers || []).map((L) => ({
      hit: new Set((L.hits || []).map((h) => Math.round(h * grid))),
      acc: new Set((L.accents || []).map((h) => Math.round(h * grid))),
      drum: L.drum || this.piano.drumForPitch(L.pitch),   // 拍の音はドラム音源で鳴らす
    }));
    const total = cols * loops;
    this._rhythmBusy = true; this._updateRhythmBtn();

    let step = 0;
    const tick = () => {
      const col = step % cols;
      (this.rhythmRows || []).forEach((cells) => cells.forEach((c, ci) => c.classList.toggle("cur", ci === col)));
      layers.forEach((L, li) => {
        if (L.hit.has(col)) {
          this.piano.playDrum(L.drum, L.acc.has(col));
          const cell = this.rhythmRows[li] && this.rhythmRows[li][col];
          if (cell) { cell.classList.add("fire"); this.animTimers.push(setTimeout(() => cell.classList.remove("fire"), cellMs * 1.2)); }
        }
      });
      step++;
      if (step >= total) { this.animTimers.push(setTimeout(() => this.stopRhythm(), cellMs)); return; }
      this.animTimers.push(setTimeout(tick, cellMs));
    };
    tick();
  }

  // 現在のキーでのコードを、読みやすい綴り {root, names} で返す
  chordSpelling(c) {
    const sharp = this.transposeRoot(c.root);
    const flat = SHARP_TO_FLAT[sharp];
    let root = sharp;
    if (flat && accCount(spellChord(flat, c.type, 4)) < accCount(spellChord(sharp, c.type, 4))) root = flat;
    return { root, names: spellChord(root, c.type, 4) };
  }

  // 現在のキーで、練習コードを「和音＋コード名」の楽譜スペックにする
  practiceChordStaff() {
    return this.pSeq.map((c) => {
      const sp = this.chordSpelling(c);
      return { chord: sp.names, label: makeChordName(sp.root, c.type) };
    });
  }
  renderPracticeStaff() {
    if (!this.pStaffWrap) return;
    this.pStaffWrap.innerHTML = "";
    this.pStaffWrap.appendChild(renderStaff(this.practiceChordStaff(), {}));
  }

  // ---- 練習 ----
  buildPractice(p) {
    this.pKind = p.chords ? "chords" : p.voicings ? "voicings" : "notes";
    this.pSeq = p.chords || p.voicings || p.notes;
    this.pIndex = 0;

    // コード練習は、選んだキーでの和音を楽譜にも表示（キー変更で追従）
    if (this.pKind === "chords") {
      this.pStaffWrap = document.createElement("div");
      this.pStaffWrap.className = "visual-wrap";
      this.bodyEl.appendChild(this.pStaffWrap);
      this.renderPracticeStaff();
    }

    const box = document.createElement("div");
    box.className = "practice-box";
    box.innerHTML = `
      <p class="phase-label">✍️ 練習 — ${p.label || ""}</p>
      <div class="quiz-prompt">
        <span class="quiz-label" id="p-label">次を弾いてください</span>
        <span class="quiz-chord" id="p-target">—</span>
      </div>
      <p class="quiz-feedback" id="p-feedback"></p>
      <div class="practice-foot">
        <div class="progress"><div class="progress-fill" id="p-bar"></div></div>
        <span class="progress-text">進捗 <strong id="p-progress">0/0</strong></span>
      </div>`;
    this.bodyEl.appendChild(box);
    this.refs = {
      label: box.querySelector("#p-label"), target: box.querySelector("#p-target"),
      feedback: box.querySelector("#p-feedback"), bar: box.querySelector("#p-bar"), progress: box.querySelector("#p-progress"),
    };
    this.showPracticeStep();
  }

  voicingMidi(offs) { const b = this.piano.baseMidi(); return offs.map((o) => b + o); }

  showPracticeStep() {
    const tgt = this.pSeq[this.pIndex];
    if (this.pKind === "chords") {
      this.refs.label.textContent = "次のコードを弾いてください";
      this.refs.target.textContent = makeChordName(this.chordSpelling(tgt).root, tgt.type);
      this.piano.setHints(this.chordMidi(tgt.root, tgt.type));
    } else if (this.pKind === "voicings") {
      const v = this.voicingMidi(tgt);
      this.refs.label.textContent = "この積み方（最低音に注意）で弾いてください";
      this.refs.target.textContent = v.map((m) => NOTE_NAMES[pitchClass(m)]).join(" - ");
      this.piano.setHintsMidi(v);
    } else {
      this.refs.label.textContent = "次の音を弾いてください";
      this.refs.target.textContent = NOTE_NAMES[(this.piano.tonic + tgt) % 12];
      this.piano.setHintsMidi([this.piano.baseMidi() + tgt]);
    }
    this.refs.progress.textContent = `${this.pIndex + 1}/${this.pSeq.length}`;
    this.refs.bar.style.width = `${(this.pIndex / this.pSeq.length) * 100}%`;
    this.refs.feedback.textContent = "";
    this.refs.feedback.className = "quiz-feedback";
  }

  update(notes) {
    if (!this.pSeq || this.locked) { this.prevActive = new Set(notes); return; }
    if (this.pKind === "chords") {
      const tgt = this.pSeq[this.pIndex];
      if (notes.length < CHORD_TYPES[tgt.type].intervals.length) return;
      if (!matchesChord(notes, this.piano.baseMidi() + NOTE_NAMES.indexOf(tgt.root), tgt.type)) return;
      this.locked = true; this.advance(true);
    } else if (this.pKind === "voicings") {
      const target = [...this.pSeq[this.pIndex]].sort((a, b) => a - b);
      const played = [...new Set(notes.map((m) => m - this.piano.baseMidi()))].sort((a, b) => a - b);
      if (target.length !== played.length || !target.every((v, i) => v === played[i])) return;
      this.locked = true; this.advance(true);
    } else {
      const expPc = (this.piano.tonic + this.pSeq[this.pIndex]) % 12;
      const newly = notes.filter((m) => !this.prevActive.has(m));
      this.prevActive = new Set(notes);
      if (newly.some((m) => pitchClass(m) === expPc)) this.advance(false);
    }
  }

  advance(lock) {
    this.pIndex += 1;
    if (this.pIndex >= this.pSeq.length) {
      this.refs.bar.style.width = "100%";
      this.refs.target.textContent = "🎉";
      this.refs.label.textContent = "できました！";
      this.refs.feedback.textContent = "クリア！";
      this.refs.feedback.className = "quiz-feedback correct";
      this.piano.setHints([]);
      this.completed.add(this.topic.title); this.save(); this.renderList();
      return;
    }
    this.refs.feedback.textContent = "✅ いいね！";
    this.refs.feedback.className = "quiz-feedback correct";
    if (lock) this.timers.push(setTimeout(() => { this.locked = false; this.showPracticeStep(); }, 550));
    else this.showPracticeStep();
  }

  // キー変更時：練習中なら現在の調で出し直し、解説なら鍵盤ハイライトを更新（調自体は変えない）
  onTranspose() {
    if (this.pSeq) {
      this.showPracticeStep();
      if (this.pKind === "chords") this.renderPracticeStaff();
    } else {
      this.startAnim();
    }
  }

  enter() { this.renderList(); this.showSlide(); }
  leave() { this.clearTimers(); this.clearAnim(); this.piano.setHints([]); }
}
