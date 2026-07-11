// compose.js — AI作曲（自動生成）。カリキュラムとは独立した3つ目の機能。
// 「モデル」＝音楽理論に基づく確率生成：
//   ・コード進行：ダイアトニック上の機能和声マルコフ連鎖（T→S→D→T の傾向）
//   ・メロディ ：コードトーン＋スケールの規則生成（強拍はコードトーン、弱拍は順次進行）

// 機能和声に基づくコード遷移の重み（度数 0=I … 6=vii°）
const CHORD_TRANS = {
  0: { 3: 3, 4: 3, 5: 2, 1: 2, 2: 1 },   // I  → IV, V, vi, ii, iii
  1: { 4: 4, 6: 1, 3: 1, 0: 1 },         // ii → V(強)
  2: { 5: 3, 3: 2, 1: 1 },               // iii→ vi, IV, ii
  3: { 4: 4, 0: 2, 1: 1, 5: 1 },         // IV → V(強), I
  4: { 0: 5, 5: 2, 3: 1 },               // V  → I(強), vi(偽終止)
  5: { 3: 3, 1: 2, 4: 1, 2: 1 },         // vi → IV, ii
  6: { 0: 4, 2: 1 },                     // vii°→ I
};

// リズム型（4拍＝1小節）。数値＝音符の拍数、{r:拍数}＝休符。各行の合計は4拍。
const RHYTHMS = [[1, 1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 2], [4], [3, 1], [1, 2, 1], [2, 1, 0.5, 0.5]];
// 休符入りのリズム型（間（ま）を作る）
const RHYTHMS_REST = [
  [1, 1, { r: 1 }, 1], [2, { r: 1 }, 1], [1, 1, 1, { r: 1 }],
  [{ r: 1 }, 1, 1, 1], [2, { r: 2 }], [1, { r: 0.5 }, 0.5, 1, 1], [1.5, { r: 0.5 }, 2],
];
const RHYTHMS_DENSE = [[1, 0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1, 1], [1, 1, 0.5, 0.5, 1], [0.5, 0.5, 0.5, 0.5, 1, 1], [1, 0.5, 0.5, 0.5, 0.5, 1], [1, 0.5, 0.5, { r: 0.5 }, 0.5, 1]];

// 各キー（長調）の綴り。調号表示と音符の異名同音を正しく出すための主音綴り。
const MAJOR_KEY = {
  0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
  6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
};
const CMP_LETTERS = ["C", "D", "E", "F", "G", "A", "B"];
const CMP_LETTER_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

// ダイアトニック音を調に合わせて綴る（例: Bb調の A#→Bb、F#調の F→E#）。
// 調号と組み合わせると VexFlow 側で余計な臨時記号を出さずに正しく表示できる。
function spellInKey(midi, tonicPc) {
  const pc = ((midi % 12) + 12) % 12;
  const deg = MAJOR_SCALE.indexOf((((pc - tonicPc) % 12) + 12) % 12);
  if (deg < 0) return midiToName(midi);                 // 非ダイアトニックはそのまま（保険）
  const tonicLetter = MAJOR_KEY[tonicPc][0];
  const letter = CMP_LETTERS[(CMP_LETTERS.indexOf(tonicLetter) + deg) % 7];
  const naturalPc = CMP_LETTER_PC[letter];
  let diff = (((pc - naturalPc) % 12) + 12) % 12; if (diff > 6) diff -= 12;   // -6..6
  const octave = Math.round((midi - diff - naturalPc) / 12) - 1;
  const acc = diff > 0 ? "#".repeat(diff) : diff < 0 ? "b".repeat(-diff) : "";
  return letter + acc + octave;
}

function cmpPick(weights) {
  const entries = Object.entries(weights);
  let total = 0; for (const [, w] of entries) total += w;
  let r = Math.random() * total;
  for (const [k, w] of entries) { r -= w; if (r <= 0) return parseInt(k, 10); }
  return parseInt(entries[0][0], 10);
}
function cmpItem(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

// コード進行（度数列）を生成。最後はI、その手前はドミナント/サブドミナントで終止感を出す。
function genProgression(bars) {
  const seq = [0];
  for (let i = 1; i < bars; i++) seq.push(cmpPick(CHORD_TRANS[seq[i - 1]]));
  seq[bars - 1] = 0;
  if (bars >= 2 && ![4, 3, 6, 1].includes(seq[bars - 2])) seq[bars - 2] = 4;
  return seq;
}

// 1小節分のメロディを生成
function genMelodyBar(scaleNotes, chordPcs, startIdx, dense) {
  const chordIdx = scaleNotes.map((_, i) => i).filter((i) => chordPcs.has(pitchClass(scaleNotes[i])));
  // 通常リズムの一部を休符入りリズムに置き換えて「間」を作る
  const pool = dense ? RHYTHMS_DENSE : (Math.random() < 0.35 ? RHYTHMS_REST : RHYTHMS);
  const rhythm = cmpItem(pool);
  let cur = startIdx;
  const notes = [];
  let beat = 0;
  for (const step of rhythm) {
    const isRest = typeof step === "object";
    const d = isRest ? step.r : step;
    if (isRest) {                                     // 休符：音は鳴らさず時間だけ進める
      notes.push({ rest: true, beats: d });
      beat += d;
      continue;
    }
    const onStrong = Math.abs(beat - Math.round(beat)) < 0.01 && Math.round(beat) % 2 === 0;
    if ((onStrong || Math.random() < 0.35) && chordIdx.length) {
      // 近いコードトーンに寄せる（少しランダム）
      const sorted = chordIdx.slice().sort((a, b) => Math.abs(a - cur) - Math.abs(b - cur));
      cur = cmpItem(sorted.slice(0, Math.min(2, sorted.length)));
    } else {
      cur += cmpItem([-2, -1, -1, 1, 1, 2]);          // 順次進行中心
      cur = Math.max(0, Math.min(scaleNotes.length - 1, cur));
    }
    notes.push({ midi: scaleNotes[cur], beats: d });
    beat += d;
  }
  return { notes, endIdx: cur };
}

const CMP_STYLES = {
  pop:     { label: "ポップ（三和音）", seventh: false, dense: false },
  jazz:    { label: "ジャズ（7th）",    seventh: true,  dense: true },
  ballad:  { label: "しっとり（三和音）", seventh: false, dense: false },
};
const CMP_TEMPOS = { slow: 80, mid: 112, fast: 140 };
// 生成の多様性（温度）。低いほど無難、高いほど大胆。学習モデル使用時のみ有効。
const CMP_TEMPS = { safe: 0.6, normal: 0.9, bold: 1.25 };

// ジャンル（学習モデルが対応していれば選べる）。表示名と、伴奏アレンジャーのスタイル対応。
const GENRE_LABELS = { pop: "ポップ", rock: "ロック", jazz: "ジャズ", folk: "フォーク", classical: "クラシック" };
const GENRE_TO_STYLE = { pop: "pop", rock: "pop", jazz: "jazz", folk: "pop", classical: "ballad" };

// メロディが極端な音域に飛ばないよう安全域にクランプ
function clampMidi(midi) { return Math.max(48, Math.min(96, midi)); }

// コード表示名（分数コードはベースを付ける。例 "Fmaj7/G"）
function chordDisplayName(bc) {
  const n = makeChordName(bc.rootName, bc.type);
  return bc.bassPc != null ? n + "/" + NOTE_NAMES[bc.bassPc] : n;
}

// クリーンな伴奏を作る。拍ごとのコードに追従し、濁らないよう最小限の声部で。
//   mode="block" … コードの和音を鳴らす（拍ごとに積む）
//   mode="arp"   … コードトーンを1音ずつアルペジオ（8分）
//   energy      … セクションの役割（"high"=サビ: 厚く / "low"=Aメロ: 薄く / "mid"=通常）
// 返り値: [{at:拍, midis:[…], dur:拍}]
function cleanAccompBar(bar, mode, energy) {
  const events = [];
  let k = 0;
  while (k < 4) {
    const bc = bar.beats[k];
    let e = k + 1; while (e < 4 && bar.beats[e].roman === bc.roman) e++;   // 同じコードの拍をまとめる
    const dur = e - k;
    let bass = 36 + (bc.bassPc != null ? bc.bassPc : bc.rootPc); if (bass < 40) bass += 12;   // 分数はそのベース
    const voicing = chooseVoicing(bc.rootPc, bc.type, 62);                  // 中音域の3〜4声
    if (mode === "arp") {
      const arp = [...voicing, voicing[0] + 12];
      events.push({ at: k, midis: [bass], dur: Math.min(dur, 2) });
      const step = energy === "low" ? 1.0 : 0.5;                            // Aメロは4分の粗いアルペジオ
      for (let i = 0; i < dur / step; i++) events.push({ at: k + i * step, midis: [arp[i % arp.length]], dur: step });
      if (energy === "high") events.push({ at: k, midis: voicing, dur: Math.min(dur, 2) });   // サビは頭に和音も
    } else {                                                                // block（和音）
      events.push({ at: k, midis: [bass], dur });
      events.push({ at: k, midis: voicing, dur });
      if (energy === "high") events.push({ at: k, midis: [voicing[voicing.length - 1] + 12], dur });  // 上声を重ねる
    }
    k = e;
  }
  return events;
}

// =================================================================
// 伴奏アレンジ（ベタ弾き→分散和音/コンピング＋ベース＋声部進行）
// =================================================================
function chordPcSet(chordMidi) { return new Set(chordMidi.map((m) => ((m % 12) + 12) % 12)); }

// 直前の和音に近い声部（転回形＋オクターブ）を選び、スムーズな声部進行にする。
function chooseVoicing(rootPc, type, prevMean) {
  const invs = chordInversions(48 + rootPc, type);   // 基本形＋各転回形
  let best = null, bestD = Infinity;
  for (const inv of invs) for (const oct of [-12, 0, 12]) {
    const v = inv.map((n) => n + oct);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    if (mean < 52 || mean > 74) continue;             // 中音域に収める
    const d = Math.abs(mean - prevMean);
    if (d < bestD) { bestD = d; best = v; }
  }
  return (best || chordNotes(48 + rootPc, type)).slice().sort((a, b) => a - b);
}

// 学習した伴奏イベントから、その小節の「ボイシング（積む音）」を推定する。
// リズムは一貫したテンプレートで統一しつつ、どの音を重ねるかはデータ由来にするための橋渡し。
function voicingFromComp(events) {
  if (!events || !events.length) return null;
  const durByMidi = new Map();
  events.forEach((e) => { const m = e.midis[0]; durByMidi.set(m, (durByMidi.get(m) || 0) + e.dur); });
  const midis = [...durByMidi.keys()].sort((a, b) => a - b);
  if (midis.length < 2) return null;
  const bass = midis[0];
  // ベースより上で、鳴っている時間が長い音を最大4声（＝主要な和声）
  const upper = midis.filter((m) => m > bass + 2)
    .sort((a, b) => (durByMidi.get(b) - durByMidi.get(a)) || a - b)
    .slice(0, 4).sort((a, b) => a - b);
  return upper.length >= 2 ? upper : midis.slice(0, 3);
}

// 1小節の伴奏イベント列 [{at:拍, midis:[…], dur:拍}] を作る。スタイルで表情を変える。
// voicingOverride を渡すと、そのボイシング（学習モデル由来など）を使う。
function arrangeBar(rootPc, type, styleKey, prevMean, voicingOverride) {
  const voicing = voicingOverride || chooseVoicing(rootPc, type, prevMean);
  const mean = voicing.reduce((a, b) => a + b, 0) / voicing.length;
  let bass = 36 + rootPc; if (bass < 40) bass += 12;                  // ルートのベース
  let bassF = 36 + ((rootPc + 7) % 12); if (bassF < 40) bassF += 12;  // 5度のベース
  const ev = [];
  if (styleKey === "ballad") {
    // 持続する和音＋低音（ゆったり）。頭でルート、途中で分散して動きを少し。
    ev.push({ at: 0, midis: [bass], dur: 3.9 });
    ev.push({ at: 0, midis: voicing, dur: 3.9 });
    ev.push({ at: 2, midis: [voicing[voicing.length - 1] + 12], dur: 2 });
  } else if (styleKey === "jazz") {
    // ベース＋シンコペーションのコンピング（7thの声部）。
    ev.push({ at: 0, midis: [bass], dur: 1.4 });
    ev.push({ at: 2, midis: [bassF], dur: 1.4 });
    ev.push({ at: 1, midis: voicing, dur: 0.5 });
    ev.push({ at: 2.5, midis: voicing, dur: 0.5 });
    ev.push({ at: 3.5, midis: voicing, dur: 0.5 });
  } else {
    // ポップ：ルート/5度のベース＋8分の分散和音（アルペジオ）。
    ev.push({ at: 0, midis: [bass], dur: 2 });
    ev.push({ at: 2, midis: [bassF], dur: 2 });
    const arp = [...voicing, voicing[0] + 12];         // 上に1オクターブ足してきらめきを
    [0.5, 1, 1.5, 2.5, 3, 3.5].forEach((b, i) => ev.push({ at: b, midis: [arp[i % arp.length]], dur: 0.5 }));
  }
  return { events: ev, mean };
}

// メロディ音の下にコードトーンのハモリを足して和音化する（強拍・ロング音を中心に）。
function harmonyBelow(midi, pcs) {
  for (let d = 3; d <= 9; d++) { const h = midi - d; if (h >= 48 && pcs.has(((h % 12) + 12) % 12)) return h; }
  return null;
}
function harmonizeBar(bar, boost) {
  const k = boost || 1;                                      // セクションによるハモリ量の増減
  let beat = 0;
  for (const nt of bar.melody) {
    if (!nt.rest) {
      const bc = bar.beats[Math.min(3, Math.floor(beat))];   // その音の拍のコード
      const pcs = chordPcSet(bc.chordMidi);
      const M = nt.midis[0];
      const strong = Math.abs(beat - Math.round(beat)) < 0.01 && Math.round(beat) % 2 === 0;
      // ロング音は高確率、強拍の中くらいは中確率、その他の8分以上も控えめに和音化
      let prob = nt.beats >= 1 ? 0.6 : (strong && nt.beats >= 0.5 ? 0.5 : (nt.beats >= 0.5 ? 0.22 : 0));
      prob = Math.min(1, prob * k);
      if (prob && Math.random() < prob) {
        const h = harmonyBelow(M, pcs);
        if (h != null) {
          nt.midis = [M, h];
          if ((strong || nt.beats >= 2) && Math.random() < 0.4) {   // 長い強拍は3声まで
            const h2 = harmonyBelow(h, pcs);
            if (h2 != null) nt.midis = [M, h, h2];
          }
        }
      }
    }
    beat += nt.beats;
  }
}

// =================================================================
// AI作曲モード
// =================================================================
class ComposeMode {
  constructor(piano) {
    this.piano = piano;
    this.keySel = document.getElementById("cmp-key");
    this.chordsEl = document.getElementById("cmp-chords");
    this.staffEl = document.getElementById("cmp-staff");
    this.infoEl = document.getElementById("cmp-info");
    this.chipEls = [];
    this.bars = null;
    this.timers = [];
    this.playing = false;

    for (let pc = 0; pc < 12; pc++) {
      const o = document.createElement("option"); o.value = pc; o.textContent = MAJOR_KEY[pc] + " 長調"; this.keySel.appendChild(o);
    }

    // テンポ：ユーザーが自由に設定（再生時に反映。生成には影響しない）。
    this.tempoSel = document.getElementById("cmp-tempo");
    this.bpmLabel = document.getElementById("cmp-bpm-label");
    if (this.tempoSel) this.tempoSel.addEventListener("input", () => { if (this.bpmLabel) this.bpmLabel.textContent = this.tempoSel.value; });

    // 伴奏：なし／和音／アルペジオ（コードから直接付けるクリーンな伴奏）
    this.accompSel = document.getElementById("cmp-accomp");
    if (this.accompSel) {
      [["arp", "アルペジオ"], ["block", "和音"], ["none", "なし"]].forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; this.accompSel.appendChild(o); });
      this.accompSel.value = "arp";
    }
    this.accompOn = true;

    // 学習済みモデル（model.js）があれば読み込む。無ければルールベースにフォールバック。
    this.net = (typeof NeuralComposer !== "undefined" && NeuralComposer.available())
      ? new NeuralComposer(window.COMPOSER_MODEL) : null;

    document.getElementById("cmp-generate").addEventListener("click", () => this.generate());
    document.getElementById("cmp-play").addEventListener("click", () => this.play());
    document.getElementById("cmp-stop").addEventListener("click", () => this.stop());
    const toEd = document.getElementById("cmp-to-editor");
    if (toEd) toEd.addEventListener("click", () => this.sendToEditor());
  }

  // 現在の生成曲を楽譜エディタへ送る（ライブラリに保存してエディタで開く）。
  // メロディ＝上段（コード名ラベル付き）、伴奏＝下段（楽譜表示と同じ音符化）でエクスポート。
  sendToEditor() {
    if (!this.bars || typeof ScoreLibrary === "undefined") return;
    const treble = [];
    let lastLabel = null;
    this.bars.forEach((bar) => {
      let beat = 0;
      bar.melody.forEach((nt) => {
        if (nt.rest) treble.push({ rest: true, beats: nt.beats });
        else {
          const e = { midis: nt.midis.slice(), beats: nt.beats };
          const lbl = chordDisplayName(bar.beats[Math.min(3, Math.floor(beat))]);
          if (lbl !== lastLabel) { e.label = lbl; lastLabel = lbl; }
          treble.push(e);
        }
        beat += nt.beats;
      });
    });
    const bass = [];
    if (this.accompOn) {
      this.bars.forEach((bar) => this.accompToBassNotes(bar).forEach((n) =>
        bass.push(n.midis ? { midis: n.midis.slice(), beats: n.beats } : { rest: true, beats: n.beats })));
    }
    const keyName = this.minorMode
      ? `${NOTE_NAMES[(this.tonicPc + 9) % 12]}マイナー`
      : `${MAJOR_KEY[this.tonicPc]}メジャー`;
    const t = new Date();
    const song = ScoreLibrary.save({
      title: `AI作曲 ${keyName} ${this.bars.length}小節 (${t.getHours()}:${String(t.getMinutes()).padStart(2, "0")})`,
      tonicPc: this.tonicPc,
      bpm: this.currentBpm(),
      treble, bass,
    });
    this.stop();
    if (window.appOpenInEditor) window.appOpenInEditor(song.id);
  }

  clearTimers() { this.timers.forEach((t) => clearTimeout(t)); this.timers = []; }

  // 再生テンポ（BPM）はユーザー設定のスライダーから読む。
  currentBpm() { return this.tempoSel ? parseInt(this.tempoSel.value, 10) || 112 : 112; }

  // 生成
  generate() {
    this.stop();
    const tonicPc = parseInt(this.keySel.value, 10);
    this.tonicPc = tonicPc;
    const temp = 0.9;                                   // 多様性の指定は廃止（固定）
    const base = 60 + tonicPc;
    this.genre = null;                                  // ジャンル指定は廃止（全データまとめて学習）
    const styleKey = "pop";                             // 伴奏はルールフォールバック時のみ使用
    const style = CMP_STYLES.pop;

    // 曲全体の進行・メロディをモデルに生成させる。長さもモデル任せ（EOSまで＝可変長）。
    this.bars = this.buildPhrase(tonicPc, base, temp, styleKey, style);
    // モデルが生成した進行の中の「反復」を検出し、その区間のメロディを再利用してモチーフを再帰。
    if (this.net && this.bars.length >= 8) this.reuseMotifs(this.bars, temp, styleKey);
    // セクションの役割（サビ/Aメロ）を推定してエネルギーを割り当て → 伴奏・ハモリ・強弱に反映
    this.applySections(this.bars);
    this.applyAccompaniment(this.bars);
    this.annotateExpression(this.bars);                 // 強弱・スラー等を事前計算（演奏と楽譜で共用）
    this.form = this.describeStructure(this.bars);
    this.minorMode = this.detectMinor(this.bars);

    // 下部ピアノを同じキーへ（ハイライト位置をそろえる）
    this.piano.setTonic(tonicPc);
    const tsel = document.getElementById("tonic"); if (tsel) tsel.value = String(tonicPc);

    this.render();
    const base2 = this.net ? "🧠 学習モデルで生成（メロディ）" : "📐 ルールで生成";
    const keyName = this.minorMode ? `${NOTE_NAMES[(tonicPc + 9) % 12]}マイナー（短調）` : `${NOTE_NAMES[tonicPc]}メジャー（長調）`;
    const tag = this.form ? `${base2}・${this.bars.length}小節・${keyName}・構造 ${this.form}` : `${base2}・${this.bars.length}小節・${keyName}`;
    this.infoEl.textContent = `${tag}。▶ 再生 で聴けます。テンポは自由に調整できます。🎲 生成 で作り直し。`;
  }

  // 曲を生成する（学習モデルは可変長＝EOSまで）。伴奏・ハモリは generate() 側で
  // セクション推定の後に付ける（サビ/Aメロで厚みを変えるため）。
  buildPhrase(tonicPc, base, temp, styleKey, style) {
    return this.net
      ? this.genNeural(tonicPc, base, temp)
      : this.genRule(tonicPc, base, 8, style);          // フォールバックは8小節
  }

  // 伴奏とハモリを付与。「伴奏」セレクタ（なし／和音／アルペジオ）に応じてコードから直接付ける。
  // 濁らないよう最小限の声部のクリーンな伴奏（学習comp は使わない）。
  // bar.energy(セクション推定) に応じて厚み・ハモリ確率・強弱(bar.vel)を変える。
  applyAccompaniment(bars) {
    const mode = this.accompSel ? this.accompSel.value : "arp";
    this.accompOn = mode !== "none";
    bars.forEach((bar) => {
      const en = bar.energy || "mid";
      bar.accomp = this.accompOn ? cleanAccompBar(bar, mode, en) : [];
      harmonizeBar(bar, en === "high" ? 1.5 : en === "low" ? 0.7 : 1);
      // 強弱: 学習済み曲線(velZ, 曲内z値)があれば連続値で、無ければセクション3段階
      bar.vel = typeof bar.velZ === "number" && isFinite(bar.velZ)
        ? Math.max(0.55, Math.min(1, 0.8 + 0.35 * bar.velZ))
        : en === "high" ? 1.0 : en === "low" ? 0.68 : 0.85;
    });
  }

  // モデルが生成した進行の反復区間を検出し、そのメロディを最初の出現からコピーする
  // ＝モチーフの再帰。反復の末尾小節だけ作り直して「毎回同じで終わらない」ようにする。
  // （形式はテンプレートで与えず、モデルが作った構造にモチーフが従う。）
  reuseMotifs(bars, temp, styleKey) {
    const W = 4;                                        // 反復を見る窓（4小節）
    if (bars.length < 2 * W) return;
    // ファジー比較: 装飾(7th/sus/分数)を落とした基幹コードで比べる。
    // Imaj7 ≒ I、V/2 ≒ V を同一視しないと、彩り・分数対応後は反復がほぼ検出できない。
    const barKey = (b) => b.beats.map((x) => romanBaseOf(x.roman)).join(",");
    const seen = new Map();
    for (let j = 0; j + W <= bars.length; j += W) {
      const key = bars.slice(j, j + W).map(barKey).join("|");
      if (seen.has(key)) {
        const i = seen.get(key);
        for (let k = 0; k < W; k++) bars[j + k].melody = JSON.parse(JSON.stringify(bars[i + k].melody));
        const beatRomans = bars.slice(j, j + W).flatMap((b) => b.beats.map((x) => x.roman));
        this.varyBarEnding(bars[j + W - 1], beatRomans, temp);
      } else {
        seen.set(key, j);
      }
    }
  }

  // 1小節の旋律だけ作り直す（反復の終止に変化を付ける）。beatRomans=窓全体の拍コード列。
  varyBarEnding(bar, beatRomans, temp) {
    if (!this.net) return;
    const mel = this.net.generateMelody(beatRomans, this.melTemp || temp, 4, this.genre);
    const last = (mel[mel.length - 1] || []);
    const base = 60 + this.tonicPc;
    const nb = last.map((nt) => nt.rest
      ? { rest: true, beats: nt.beats }
      : { midis: [clampMidi(base + nt.off)], beats: nt.beats });
    if (nb.length) bar.melody = nb;          // ハモリ付けは applyAccompaniment がまとめて行う
  }

  // 生成された進行の反復構造にラベル付け（例: "A A B A"）。表示用。
  describeStructure(bars) {
    const W = 4;
    if (!bars || bars.length < W) return null;
    const barKey = (b) => b.beats.map((x) => romanBaseOf(x.roman)).join(",");
    const map = new Map(); let next = 0; const labels = [];
    for (let j = 0; j + W <= bars.length; j += W) {
      const key = bars.slice(j, j + W).map(barKey).join("|");
      if (!map.has(key)) map.set(key, String.fromCharCode(65 + (next++ % 26)));
      labels.push(map.get(key));
    }
    return labels.length > 1 ? labels.join(" ") : null;
  }

  // セクションの役割推定 → エネルギー(伴奏の厚み・ハモリ・強弱)の割り当て。
  // 学習済みの強弱曲線(bar.velZ)があればそれを優先（曲調に応じた盛り上げ＝データ由来）。
  // 無ければ従来の反復ヒューリスティック(最頻出4小節ブロック=サビ)にフォールバック。
  applySections(bars) {
    const zs = bars.map((b) => b.velZ).filter((z) => typeof z === "number" && isFinite(z));
    if (zs.length >= bars.length / 2) {
      const sorted = zs.slice().sort((a, b) => a - b);
      const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
      const hi = q(0.7), lo = q(0.3);
      const spread = sorted[sorted.length - 1] - sorted[0];
      bars.forEach((b) => {
        b.energy = spread < 0.25 ? "mid"                    // 起伏の乏しい曲は平坦なまま
          : b.velZ >= hi ? "high" : b.velZ <= lo ? "low" : "mid";
      });
      return;
    }
    const W = 4;
    bars.forEach((b) => { b.energy = "mid"; });
    if (!bars || bars.length < 2 * W) return;
    const keys = [];
    for (let j = 0; j + W <= bars.length; j += W)
      keys.push(bars.slice(j, j + W).map((b) => b.beats.map((x) => romanBaseOf(x.roman)).join(",")).join("|"));
    const count = new Map();
    keys.forEach((k) => count.set(k, (count.get(k) || 0) + 1));
    const maxRep = Math.max(...count.values());
    if (maxRep < 2) return;                              // 反復が無い曲は起伏を付けない
    keys.forEach((k, bi) => {
      const rep = count.get(k);
      const energy = rep === maxRep ? "high" : rep === 1 ? "low" : "mid";
      for (let i = 0; i < W; i++) if (bars[bi * W + i]) bars[bi * W + i].energy = energy;
    });
    for (let i = keys.length * W; i < bars.length; i++) bars[i].energy = bars[i - 1].energy;   // 端数は引き継ぐ
  }

  // 長調/短調の判定（データは平行長調正規化なので、短調曲は vi 中心の進行として現れる）
  detectMinor(bars) {
    return isMinorRomans(bars.flatMap((b) => b.beats.map((x) => x.roman)));
  }

  // 音符単位の表情（強弱・アーティキュレーション・スラー）を事前計算して bars に注釈する。
  // play()（音）と render()（楽譜の強弱記号・スラー・rit.）の両方がこの結果を使う＝演奏と楽譜が一致。
  annotateExpression(bars) {
    const baseVels = bars.map((b) => b.vel || 0.85);
    bars.forEach((bar, bi) => {
      const vel = baseVels[bi];
      const nextVel = bi + 1 < bars.length ? baseVels[bi + 1] : vel;
      // マクロな起伏は学習済み強弱曲線(bar.vel)が担う。無いときだけ固定のフレーズ山型
      const phraseArc = (typeof bar.velZ === "number" && isFinite(bar.velZ)) ? 0 : [0, 0.03, 0.05, -0.03][bi % 4];
      const notes = bar.melody;
      let cur = 0;
      notes.forEach((nt, ni) => {
        // --- 強弱: 小節内で次の小節のレベルへ線形補間（クレシェンド/デクレシェンドが連続になる）---
        const base = vel + (nextVel - vel) * (cur / 4);
        let v = base + 0.07 + phraseArc;
        const strong = Math.abs(cur - Math.round(cur)) < 0.01 && Math.round(cur) % 2 === 0;
        if (strong) v += 0.04;
        if (nt.beats >= 2) v += 0.03;
        if (!nt.rest) {
          const prevN = notes[ni - 1], nextN = notes[ni + 1];
          const isPeak = (!prevN || prevN.rest || nt.midis[0] > prevN.midis[0]) &&
                         (!nextN || nextN.rest || nt.midis[0] > nextN.midis[0]);
          if (isPeak) v += 0.04;
        }
        nt.vel = Math.max(0.35, Math.min(1, v));
        // --- アーティキュレーション: 順次進行はレガート(スラー)、跳躍・休符前は離す ---
        if (!nt.rest) {
          const nextN = notes[ni + 1] || ((bars[bi + 1] || {}).melody || [])[0];
          let artic = 0.92, legato = false;
          if (!nextN || nextN.rest) artic = 0.85;                          // フレーズの切れ目は離す
          else {
            const iv = Math.abs((nextN.midis ? nextN.midis[0] : nt.midis[0]) - nt.midis[0]);
            if (iv <= 2) { artic = 1.02; legato = true; }                  // スラー
            else if (iv >= 7) artic = 0.85;                                // デタッシェ
          }
          if (nt.beats >= 2) artic = Math.max(artic, 0.98);                // ロングノートは全長で歌う
          nt.artic = artic;
          nt.legatoNext = legato && ni + 1 < notes.length;                 // 楽譜のスラーは小節内のみ
        }
        cur += nt.beats;
      });
      // 小節の強弱記号（楽譜表示用）
      bar.dynMark = vel < 0.66 ? "p" : vel < 0.79 ? "mp" : vel < 0.92 ? "mf" : "f";
    });
    let last = null;                                       // 記号は変化した小節にだけ表示
    bars.forEach((bar) => { bar.showDyn = bar.dynMark !== last; last = bar.dynMark; });
    // クレシェンド/デクレシェンド（松葉）: 強弱曲線の単調な上昇/下降区間（2小節以上・変化量0.06以上）
    this.hairpins = [];
    let i = 0;
    while (i < baseVels.length - 1) {
      const dir = Math.sign(baseVels[i + 1] - baseVels[i]);
      if (!dir) { i++; continue; }
      let j = i;
      while (j + 1 < baseVels.length && Math.sign(baseVels[j + 1] - baseVels[j]) === dir
             && Math.abs(baseVels[j + 1] - baseVels[j]) > 0.004) j++;
      if (j > i && Math.abs(baseVels[j] - baseVels[i]) >= 0.06)
        this.hairpins.push({ from: i, to: j, type: dir > 0 ? "cresc" : "dim" });
      i = Math.max(j, i + 1);
    }
    // 記譜の慣習: 松葉の途中の強弱文字は出さない（両端＝開始レベルと到達レベルのみ）
    this.hairpins.forEach((h) => {
      for (let b = h.from + 1; b < h.to; b++) bars[b].showDyn = false;
      if (bars[h.to].dynMark !== bars[h.from].dynMark) bars[h.to].showDyn = true;   // 到達レベルを明示
    });
  }

  // 拍ごとの1コードを {roman,rootPc,type,rootName,bassPc,chordMidi} に変換
  beatChord(roman, tonicPc) {
    const { rootSemitone, typeKey, bassSemitone } = romanToChord(roman);
    const rootPc = (tonicPc + rootSemitone) % 12;
    const bassPc = bassSemitone != null ? (tonicPc + bassSemitone) % 12 : null;
    return { roman, rootPc, type: typeKey, rootName: NOTE_NAMES[rootPc], bassPc, chordMidi: chordNotes(48 + rootPc, typeKey) };
  }

  // 学習済みLSTMで生成。コードは「1拍1コード（保持で長いコード）」＝和声リズムをモデルが決める。
  genNeural(tonicPc, base, temp) {
    // Transformerコードは温度を下げると長距離の反復（A/Bメロ・サビ的な曲構造）が出やすい。
    // 動きとのバランスで 0.85 前後。メロディは低めにしてコードトーン追従を強める。
    const chordTemp = 0.85;
    this.melTemp = Math.max(0.4, Math.min(0.85, temp * 0.72));   // 旋律の音高変化を少し増やす
    const beatRomans = this.net.generateChordsUntilEnd(this.genre, chordTemp, 6, 32);   // 可変長（拍ごと・EOSまで）
    const barCount = Math.floor(beatRomans.length / 4);
    const melBars = this.net.generateMelody(beatRomans, this.melTemp, 4, this.genre);   // 拍文脈で旋律
    this.shapeMelodyEnding(melBars, beatRomans);         // 終止: 最後の音をトニックのロングノートへ
    // 強弱曲線: 曲調(進行とメロディの形)から「どこを盛り上げ、どこを優しく」を学習モデルで推論
    const velZ = this.net.dynamicsCurve ? this.net.dynamicsCurve(beatRomans, melBars) : null;
    const bars = [];
    for (let bi = 0; bi < barCount; bi++) {
      const beats = beatRomans.slice(bi * 4, bi * 4 + 4).map((r) => this.beatChord(r, tonicPc));
      const melody = (melBars[bi] || []).map((nt) => nt.rest
        ? { rest: true, beats: nt.beats }
        : { midis: [clampMidi(base + nt.off)], beats: nt.beats });
      bars.push({ ...beats[0], beats, melody, velZ: velZ ? velZ[bi] : null });   // トップレベルは代表（先頭拍）コード
    }
    return bars;
  }

  // メロディの終止: 最終小節の最後の音を「実際の最終コード」のコードトーンへ寄せ、
  // 後続の休符を吸収してのばす。トニック強制はしない（終止のかたちはモデル任せ）＝濁りだけ防ぐ。
  shapeMelodyEnding(melBars, beatRomans) {
    const lastMel = melBars[melBars.length - 1];
    if (!lastMel || !lastMel.length) return;
    let li = -1;
    for (let i = lastMel.length - 1; i >= 0; i--) if (!lastMel[i].rest) { li = i; break; }
    if (li < 0) return;
    const { rootSemitone, typeKey } = romanToChord(beatRomans[beatRomans.length - 1]);
    const ivs = (CHORD_TYPES[typeKey] || CHORD_TYPES.maj).intervals;
    const targets = ivs.map((iv) => (rootSemitone + iv) % 12);
    const n = lastMel[li];
    let best = n.off, bestD = 99;
    for (let c = -8; c <= 8; c++) {
      const o = n.off + c;
      if (targets.includes(((o % 12) + 12) % 12) && Math.abs(c) < bestD) { bestD = Math.abs(c); best = o; }
    }
    n.off = best;
    let tail = 0;
    for (let i = li + 1; i < lastMel.length; i++) tail += lastMel[i].beats;
    n.beats += tail;                                     // 後続休符を吸収してロングノートに
    lastMel.length = li + 1;
  }

  // ルールベース生成（学習モデルが無いときのフォールバック）
  genRule(tonicPc, base, barCount, style) {
    // メロディ用スケール音（主音C5付近から約2オクターブ）
    const scaleNotes = [];
    for (let o = 0; o < 2; o++) for (const s of MAJOR_SCALE) scaleNotes.push(base + o * 12 + s);
    scaleNotes.push(base + 24);

    const degrees = genProgression(barCount);
    const table = style.seventh ? DIATONIC_SEVENTH : DIATONIC_TRIAD;
    let melIdx = 7;  // 中ほどから開始

    return degrees.map((deg) => {
      const rootPc = (tonicPc + MAJOR_SCALE[deg]) % 12;
      const type = table[deg];
      const chordPcs = new Set(CHORD_TYPES[type].intervals.map((i) => (rootPc + i) % 12));
      const rootMidi = 48 + rootPc;                 // C3付近で伴奏
      const chordMidi = chordNotes(rootMidi, type);
      let roman = ROMAN[deg]; if (style.seventh) roman += (type === "dom7" ? "7" : type === "m7b5" ? "ø7" : "7");
      const mel = genMelodyBar(scaleNotes, chordPcs, melIdx, style.dense);
      melIdx = mel.endIdx;
      const melody = mel.notes.map((n) => n.rest ? { rest: true, beats: n.beats } : { midis: [n.midi], beats: n.beats });
      const rep = { rootName: NOTE_NAMES[rootPc], rootPc, type, roman, chordMidi };
      return { ...rep, beats: [rep, rep, rep, rep], melody };   // 1小節1コード（各拍同じ）
    });
  }

  // 1小節の伴奏イベント({at,dur,midis})を、下段譜の音符列に変換。
  // 同じ発音位置のイベントは1つの和音にまとめ、次の発音位置までを音価にする（コンピングの近似）。
  // 返り値の各要素: { midis:number[]|null(=休符), beats, atBeat }。合計4拍。
  accompToBassNotes(bar) {
    const evs = (bar.accomp || []).filter((e) => e.midis && e.midis.length);
    if (!evs.length) {                                   // 伴奏イベントが無ければ全音符の和音
      return [{ midis: bar.chordMidi.slice(), beats: 4, atBeat: 0 }];
    }
    const byAt = new Map();                              // 発音位置(16分に量子化) → 音の集合
    evs.forEach((e) => {
      const at = Math.min(3.75, Math.max(0, Math.round(e.at * 4) / 4));
      if (!byAt.has(at)) byAt.set(at, new Set());
      e.midis.forEach((m) => byAt.get(at).add(m));
    });
    const onsets = [...byAt.keys()].sort((a, b) => a - b);
    const notes = [];
    let cursor = 0;
    onsets.forEach((o, i) => {
      if (o - cursor > 0.01) notes.push({ midis: null, beats: o - cursor, atBeat: cursor });   // 発音前の休符
      const next = i + 1 < onsets.length ? onsets[i + 1] : 4;
      const dur = Math.max(0.25, next - o);
      notes.push({ midis: [...byAt.get(o)].sort((a, b) => a - b), beats: dur, atBeat: o });
      cursor = next;
    });
    if (cursor < 4 - 0.01) notes.push({ midis: null, beats: 4 - cursor, atBeat: cursor });     // 末尾の休符
    return notes;
  }

  // 連続する同じコードの拍をまとめて「セグメント」にする（コードが変わる位置＝和声リズム）。
  computeSegments() {
    const segs = [];
    this.bars.forEach((bar, bi) => bar.beats.forEach((bc, k) => {
      const last = segs[segs.length - 1];
      if (last && last.roman === bc.roman) last.beats++;
      else segs.push({ roman: bc.roman, rootName: bc.rootName, type: bc.type, bassPc: bc.bassPc, startBeat: bi * 4 + k, beats: 1 });
    }));
    return segs;
  }

  render() {
    // コード進行のチップ（コードが変わる区切りごと＝拍数もわかる）
    this.segments = this.computeSegments();
    this.chordsEl.innerHTML = "";
    this.chipEls = this.segments.map((sg) => {
      const chip = document.createElement("div");
      chip.className = "cmp-chip";
      const beatsLabel = sg.beats === 4 ? "" : `<span class="cmp-beats">${sg.beats}拍</span>`;
      chip.innerHTML = `<span class="cmp-roman">${sg.roman}</span><span class="cmp-name">${chordDisplayName(sg)}</span>${beatsLabel}`;
      this.chordsEl.appendChild(chip);
      return chip;
    });
    // メロディ＋コード名の楽譜（コードが変わる位置にコード名ラベル、音価/休符も反映）。
    // 演奏表現も記譜: 強弱記号(p/mp/mf/f, 変化した小節のみ)・スラー(レガート)・rit.(終盤)。
    const specs = [];
    let lastLabel = null;
    const barFirstGi = [];                               // 各小節の先頭エントリindex（松葉の位置決め用）
    this.bars.forEach((bar, bi) => {
      barFirstGi[bi] = specs.length;
      let beat = 0;
      bar.melody.forEach((nt, ni) => {
        let spec;
        if (nt.rest) spec = { rest: true, beats: nt.beats };
        else if (nt.midis.length > 1)                    // ハモリつき＝和音として表示
          spec = { chord: nt.midis.slice().sort((a, b) => a - b).map((m) => spellInKey(m, this.tonicPc)), beats: nt.beats };
        else spec = { name: spellInKey(nt.midis[0], this.tonicPc), beats: nt.beats };
        const bc = bar.beats[Math.min(3, Math.floor(beat))];   // この音の拍のコード
        const lbl = chordDisplayName(bc);
        if (lbl !== lastLabel) { spec.label = lbl; lastLabel = lbl; }   // コードが変わったら表示
        if (ni === 0) {
          if (bar.showDyn && bar.dynMark) spec.dyn = bar.dynMark;        // 強弱記号
          // rit. は実際にかかる曲（強弱曲線が終盤下降）にだけ表示＝演奏と一致
          if (bi === this.bars.length - 2 && this.endingRit() > 0.05) spec.mark = "rit.";
        }
        if (nt.legatoNext) spec.slurNext = true;                          // スラー（次の音へ）
        specs.push(spec);
        beat += nt.beats;
      });
    });
    // 伴奏（下段・ヘ音記号）：伴奏オンのときだけ描く。オフ時は上段（メロディ）のみ。
    let bass = null;
    this.bassSeq = [];
    if (this.accompOn) {
      this.bassByBar = this.bars.map((bar) => this.accompToBassNotes(bar));
      bass = [];
      let bgi = 0;
      this.bassByBar.forEach((notes, bi) => {
        notes.forEach((n) => {
          bass.push(n.midis
            ? { chord: n.midis.map((m) => spellInKey(m, this.tonicPc)), beats: n.beats }
            : { rest: true, beats: n.beats });
          this.bassSeq.push({ gi: bgi, bar: bi, atBeat: n.atBeat, midis: n.midis });
          bgi++;
        });
      });
    }
    this.staffEl.innerHTML = "";
    // 段あたりの小節数は renderStaff 側で「コンテナ幅に収まる数」に自動調整させる
    // （密な小節を詰め込んで縮小 → つぶれるのを防ぐ）。上限は 4 小節/段。
    const cw = this.staffEl.clientWidth || 600;
    // 松葉（クレシェンド/デクレシェンド）: 小節区間 → エントリindex区間に変換して渡す
    const hairpins = (this.hairpins || []).map((h) => ({
      fromGi: barFirstGi[h.from], toGi: barFirstGi[h.to] != null ? barFirstGi[h.to] : specs.length - 1, type: h.type,
    }));
    // 調号・拍子（4/4）・小節線（4拍ごと）＋メロディ(上段)＆伴奏(下段)の大譜表＋多段表示
    const host = renderStaff(specs, { keySig: MAJOR_KEY[this.tonicPc], timeSig: "4/4", barBeats: 4, measuresPerRow: 4, fitWidth: cw, bass, hairpins });
    this.staffEl.appendChild(host);
    this.staffHost = host;
    this._scrolledRow = -1;

    // 3段以上あるときは「2段ぶん」の高さに制限して、再生に合わせてスクロール
    if (host._rowCount > 2) {
      const svg = host.querySelector("svg");
      const rh = svg ? svg.getBoundingClientRect().height : 0;
      const perRow = rh / host._rowCount;
      this.staffEl.style.maxHeight = (perRow ? Math.round(perRow * 2 + 24) : 320) + "px";
      this.staffEl.style.overflowY = "auto";
    } else {
      this.staffEl.style.maxHeight = "";
      this.staffEl.style.overflowY = "";
    }
    this.staffEl.scrollTop = 0;
  }

  // 現在の小節が見えるよう楽譜コンテナを縦スクロール（段が変わったときだけ）
  scrollToMeasure(bi) {
    const host = this.staffHost;
    if (!host || host._rowCount <= 2) return;
    const row = host._measureRow[bi];
    if (row === this._scrolledRow) return;
    this._scrolledRow = row;
    const anchor = (host._rowAnchor && host._rowAnchor[row]) || host._measureMark[bi];
    if (!anchor || !anchor.getBoundingClientRect) return;
    const cRect = this.staffEl.getBoundingClientRect();
    const mRect = anchor.getBoundingClientRect();
    const delta = (mRect.top - cRect.top) - 14;   // その段を上端付近へ
    if (Math.abs(delta) > 2) this.staffEl.scrollBy({ top: delta, behavior: "smooth" });
  }

  // 楽譜上段の現在音をハイライト（gi=グローバル音符index、-1で消灯）
  setStaffHighlight(gi) {
    const host = this.staffHost;
    if (!host || !host._marks) return;
    host._marks.forEach((m, j) => { if (m) m.classList.toggle("on", j === gi); });
  }

  // 楽譜下段（伴奏）の現在音をハイライト
  setBassHighlight(gi) {
    const host = this.staffHost;
    if (!host || !host._bassMarks) return;
    host._bassMarks.forEach((m, j) => { if (m) m.classList.toggle("on", j === gi); });
  }

  // 鍵盤の発光を「いま鳴っているメロディ＋伴奏」の音で更新（音名＝ピッチクラスで点灯）
  refreshHints() {
    this.piano.setHints([...(this._melMidis || []), ...(this._accMidis || [])]);
  }

  highlightChip(i) {
    this.chipEls.forEach((el, j) => el.classList.toggle("current", j === i));
  }

  // 終盤リタルダンドの深さ(0=なし〜0.3)。固定ではなく、学習した強弱曲線が終盤で下降している
  // 曲（=静かに閉じる曲想）だけ、その下降量に応じてかける。強いまま終わる曲はインテンポ。
  endingRit() {
    const bars = this.bars || [];
    const n = bars.length;
    if (n < 6) return 0;
    const v = (b) => (b && typeof b.vel === "number" ? b.vel : 0.85);
    const drop = v(bars[n - 4]) - v(bars[n - 1]);
    return Math.max(0, Math.min(0.3, drop * 3));
  }

  // テンポマップ: 基本は一定テンポ。endingRit() が正の曲だけ最後の2小節を徐々にのばす。
  // 返り値: { starts: 各小節の開始ms（最後は曲の総尺）, factors: 小節ごとの伸長率, beatTime(拍→ms) }
  buildTempoMap(nBars, beatMs) {
    const rit = this.endingRit();
    const factors = [];
    for (let bi = 0; bi < nBars; bi++)
      factors.push(bi === nBars - 1 ? 1 + rit : bi === nBars - 2 ? 1 + rit * 0.35 : 1.0);
    const starts = [0];
    for (let bi = 0; bi < nBars; bi++) starts.push(starts[bi] + 4 * beatMs * factors[bi]);
    const beatTime = (gb) => {
      const bi = Math.max(0, Math.min(nBars - 1, Math.floor(gb / 4)));
      return starts[bi] + (gb - bi * 4) * beatMs * factors[bi];
    };
    return { starts, factors, beatTime };
  }

  async play() {
    if (!this.bars) this.generate();
    await this.piano.ensureAudio();
    this.stop();
    this.playing = true;
    const beatMs = 60 / this.currentBpm() * 1000;      // テンポはユーザー設定のスライダーから
    const tm = this.buildTempoMap(this.bars.length, beatMs);
    this.staffEl.scrollTop = 0;
    this._scrolledRow = -1;
    this._melMidis = [];                              // いま鳴っているメロディ音
    this._accMidis = [];                              // いま鳴っている伴奏音
    let gi = 0;                                       // 楽譜エントリのグローバルindex
    // コードチップ（セグメント）のハイライトは、コードが変わる拍のタイミングで点灯
    (this.segments || []).forEach((sg, si) => {
      this.timers.push(setTimeout(() => { if (this.playing) this.highlightChip(si); }, tm.beatTime(sg.startBeat)));
    });
    this.bars.forEach((bar, bi) => {
      const t = tm.starts[bi], f = tm.factors[bi];
      // 小節頭：スクロール追従
      this.timers.push(setTimeout(() => {
        if (!this.playing) return;
        this.scrollToMeasure(bi);                     // 現在小節の段へスクロール
      }, t));
      // 伴奏の音声（アルペジオ/コンピング＋ベース）。セクションのエネルギーで強弱をつける
      const vel = bar.vel || 0.85;
      (bar.accomp || []).forEach((evn) => {
        const accent = evn.at === 0 ? 0.06 : 0;       // 小節頭のベース/和音は少し強く
        this.timers.push(setTimeout(() => {
          if (this.playing) this.piano.playTones(evn.midis.map(midiToName), (evn.dur * beatMs * f / 1000) * 0.95, Math.min(1, vel * 0.9 + accent));
        }, t + evn.at * beatMs * f));
      });
      // メロディ: annotateExpression が事前計算した強弱(nt.vel)・アーティキュレーション(nt.artic)を使う
      // ＝楽譜の強弱記号・スラー表示と完全に一致した演奏になる。
      let cur = 0;
      bar.melody.forEach((nt) => {
        const g = gi;                                 // この音符の楽譜index
        const mt = t + cur * beatMs * f;
        const v = nt.vel != null ? nt.vel : Math.min(1, vel + 0.07);
        const artic = nt.artic != null ? nt.artic : 0.9;
        this.timers.push(setTimeout(() => {
          if (!this.playing) return;
          this.setStaffHighlight(g);                  // 上段でこの音を光らせる
          this._melMidis = nt.rest ? [] : nt.midis;   // 鍵盤：メロディの現在音を更新
          this.refreshHints();
          if (!nt.rest) this.piano.playTones(nt.midis.map(midiToName), (nt.beats * beatMs * f / 1000) * artic, v);
        }, mt));
        cur += nt.beats;
        gi++;
      });
    });
    // 下段（伴奏）の楽譜ハイライト＋鍵盤発光を、各音符の発音位置に合わせて追従
    (this.bassSeq || []).forEach((bn) => {
      const when = tm.beatTime(bn.bar * 4 + bn.atBeat);
      this.timers.push(setTimeout(() => {
        if (!this.playing) return;
        this.setBassHighlight(bn.gi);                 // 下段でこの音を光らせる
        this._accMidis = bn.midis || [];              // 鍵盤：伴奏の現在音を更新
        this.refreshHints();
      }, when));
    });
    this.timers.push(setTimeout(() => {
      this.playing = false;
      this._melMidis = []; this._accMidis = [];
      this.piano.setHints([]); this.highlightChip(-1);
      this.setStaffHighlight(-1); this.setBassHighlight(-1);
      this.clearPlayCursor();
    }, tm.starts[this.bars.length] + 60));
    // 再生カーソル：テンポマップから「経過時間→小節＋小節内進捗」を毎フレーム求めて動かす。
    // 小節内は等速（rit. も小節単位の伸長）なので frac は時間比でそのまま拍進捗になる。
    const t0 = performance.now();
    const totalMs = tm.starts[this.bars.length];
    const tick = () => {
      if (!this.playing) return;
      const el = performance.now() - t0;
      if (el >= totalMs) { this.clearPlayCursor(); return; }
      let bi = 0;
      while (bi + 1 < this.bars.length && tm.starts[bi + 1] <= el) bi++;
      const frac = (el - tm.starts[bi]) / Math.max(1, tm.starts[bi + 1] - tm.starts[bi]);
      if (this.staffHost && this.staffHost.setPlayCursor) this.staffHost.setPlayCursor(bi, frac);
      this._cursorRaf = requestAnimationFrame(tick);
    };
    this._cursorRaf = requestAnimationFrame(tick);
  }

  // 再生カーソルを消す（再生終了・停止時）
  clearPlayCursor() {
    if (this._cursorRaf) { cancelAnimationFrame(this._cursorRaf); this._cursorRaf = 0; }
    const host = this.staffHost;
    if (host && host.setPlayCursor) host.setPlayCursor(-1);
  }

  stop() {
    this.clearTimers();
    this.playing = false;
    this._melMidis = []; this._accMidis = [];
    this.piano.setHints([]);
    this.highlightChip(-1);
    this.setStaffHighlight(-1);
    this.setBassHighlight(-1);
    this.clearPlayCursor();
  }

  update() {}
  // ウィンドウ幅の変化で楽譜の段組みを引き直す（再生中はハイライト参照が壊れるため見送る）
  onResize() {
    if (this.bars && !this.playing) this.render();
  }
  enter() { if (!this.bars) this.generate(); }
  leave() { this.stop(); }
}
