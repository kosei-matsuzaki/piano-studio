// visuals.js — 楽譜（五線譜）の描画。VexFlow を使用して正しい記譜で表示する。
// 鍵盤の図は画面下部のピアノを使うため、ここでは楽譜のみを扱う。

const SOLFEGE = { C: "ド", D: "レ", E: "ミ", F: "ファ", G: "ソ", A: "ラ", B: "シ" };

// "C4","F#4","Bb3" -> { key:"c#/4", letter:"C", acc:"#"|"b"|null }
function parseNote(name) {
  const m = String(name).match(/^([A-G])(##|bb|#|b|)(-?\d+)$/);
  if (!m) return { key: "c/4", letter: "C", acc: null };
  const acc = m[2] || null;
  return { key: m[1].toLowerCase() + (acc || "") + "/" + m[3], letter: m[1], acc };
}

function accMark(acc) { return acc === "#" ? "♯" : acc === "b" ? "♭" : ""; }

// 全音階インデックス（音高の上下を測る用）。C4=28, E4=30, F5=38。
const LETTER_NUM = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
function diOf(name) {
  const m = String(name).match(/^([A-G])(?:##|bb|#|b|)(-?\d+)$/);
  return m ? parseInt(m[2], 10) * 7 + LETTER_NUM[m[1]] : 30;
}

// 拍数（1拍＝4分音符）→ VexFlow の音価。近い値に丸め、付点も返す。
//   4=全, 3=付点2分, 2=2分, 1.5=付点4分, 1=4分, 0.75=付点8分, 0.5=8分, 0.25=16分
const DUR_TABLE = [
  [4, "w", 0], [3, "h", 1], [2, "h", 0], [1.5, "q", 1],
  [1, "q", 0], [0.75, "8", 1], [0.5, "8", 0], [0.25, "16", 0],
];
function beatsToDur(beats) {
  const b = beats == null ? 4 : beats;
  let best = DUR_TABLE[0];
  for (const row of DUR_TABLE) if (Math.abs(row[0] - b) < Math.abs(best[0] - b)) best = row;
  return { duration: best[1], dots: best[2] };
}

// 譜面で正確に表せる音価（16分=1スロット単位）。全, 付点2分, 2分, 付点4分, 4分, 付点8分, 8分, 16分。
const REP_SLOTS = [16, 12, 8, 6, 4, 3, 2, 1];
// スロット数 n を、表せる音価の並びに貪欲分解（例: 5→[4,1], 13→[12,1]）。
function decomposeSlots(n) {
  const out = [];
  let r = Math.max(1, Math.round(n));
  while (r > 0) { let took = 1; for (const s of REP_SLOTS) if (s <= r) { took = s; break; } out.push(took); r -= took; }
  return out;
}
// 1小節分のエントリを、表せる音価の並びに整える（合計が barBeats*4 スロットになるよう末尾で調整）。
// 返り値: [{ base:元エントリ, beats, tieStart, tieEnd }]（tie=同じ音符の分割を後でつなぐ）。
function tileMeasure(measure, barBeats) {
  const items = measure.map((e) => ({ e, slots: Math.max(1, Math.round((e.beats == null ? 4 : e.beats) * 4)) }));
  if (barBeats) {                                            // 小節拍数が決まっているときだけ合計を合わせる
    const total = Math.round(barBeats * 4);
    let d = total - items.reduce((s, it) => s + it.slots, 0);
    for (let i = items.length - 1; i >= 0 && d !== 0; i--) { // 差分を末尾から吸収（1未満にはしない）
      const adj = d < 0 ? Math.max(d, -(items[i].slots - 1)) : d;
      items[i].slots += adj; d -= adj;
    }
  }
  const out = [];
  items.forEach(({ e, slots }) => {
    const pieces = decomposeSlots(slots);
    pieces.forEach((ps, idx) => out.push({ base: e, beats: ps * 0.25, tieStart: idx > 0, tieEnd: idx < pieces.length - 1 }));
  });
  return out;
}

// 楽譜の1エントリを {names:[音名...], beats, rest, label} に正規化する。
//   "C4"                          … 単音（全音符）
//   ["C4","E4","G4"]              … 和音（縦に積む）
//   {name:"C4", beats:1, label}   … 単音＋音価（拍）＋ラベル
//   {chord:["C4","E4","G4"], beats, label} … 和音＋ラベル
//   {chord:{root:"C",type:"maj",oct:4}, label} … コードを正しい綴りで和音化
//   {rest:true, beats:1}          … 休符
// 演奏表現の記譜（任意）: dyn="p/mp/mf/f"（強弱記号）, mark="rit."等（発想標語）, slurNext=次の音へスラー
function normalizeEntry(e) {
  if (typeof e === "string") return { names: [e], beats: 4 };
  if (Array.isArray(e)) return { names: e, beats: 4 };
  const beats = e.beats == null ? 4 : e.beats;
  const expr = { dyn: e.dyn, mark: e.mark, slurNext: e.slurNext };
  if (e.rest) return { names: [], rest: true, beats, label: e.label, ...expr };
  if (e.chord) {
    if (Array.isArray(e.chord)) return { names: e.chord, label: e.label, beats, ...expr };
    return { names: spellChord(e.chord.root, e.chord.type, e.chord.oct), label: e.label, beats, ...expr };
  }
  return { names: [e.name], label: e.label, beats, ...expr };
}

const SVG_NS = "http://www.w3.org/2000/svg";
function mkSvg(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

// 調号のシャープ/フラット数（横幅の見積り用）
const KEYSIG_COUNT = { C: 0, G: 1, D: 2, A: 3, E: 4, B: 5, "F#": 6, "C#": 7, F: 1, Bb: 2, Eb: 3, Ab: 4, Db: 5, Gb: 6, Cb: 7 };

// 1エントリ（音符/和音/休符）を VexFlow の StaveNote に変換。clef は "treble"|"bass"。
function makeStaveNote(e, opts, VF, clef) {
  clef = clef || "treble";
  const { duration, dots } = beatsToDur(e.beats);
  // 付点は duration 文字列に含めないと tick（内部の長さ）に反映されず、
  // 上下段の小節線がずれる。Dot.buildAndAttach は見た目の点の描画のみ。
  const durStr = duration + "d".repeat(dots);
  if (e.rest) {
    const sn = new VF.StaveNote({ keys: [clef === "bass" ? "d/3" : "b/4"], duration: durStr + "r", clef });
    if (dots) VF.Dot.buildAndAttach([sn], { all: true });
    return sn;
  }
  const parsed = e.names.map(parseNote);
  const sn = new VF.StaveNote({ keys: parsed.map((p) => p.key), duration: durStr, clef });
  // 符尾の向き：中央線より上の音は下向き、下の音は上向き（treble=B4=di34, bass=D3=di22）。
  const pivot = clef === "bass" ? 22 : 34;
  const dis = e.names.map(diOf);
  const hi = Math.max(...dis), lo = Math.min(...dis);
  sn.setStemDirection((hi - pivot) >= (pivot - lo) ? -1 : 1);
  // 調号がある場合は applyAccidentals に任せる（調号内の音に♯♭を重複表示しない）
  if (!opts.keySig) parsed.forEach((p, idx) => { if (p.acc) sn.addModifier(new VF.Accidental(p.acc), idx); });
  if (dots) VF.Dot.buildAndAttach([sn], { all: true });
  return sn;
}

// エントリ列を小節（barBeats 拍ごと）に分割。barBeats 未指定なら全体で1小節。
function splitMeasures(entries, barBeats) {
  if (!barBeats) return [entries.slice()];
  const measures = []; let cur = []; let acc = 0;
  entries.forEach((e) => {
    cur.push(e);
    acc += e.beats == null ? 4 : e.beats;
    if (acc >= barBeats - 0.001) { measures.push(cur); cur = []; acc = 0; }
  });
  if (cur.length) measures.push(cur);
  return measures;
}

// 五線譜を描画。noteSpecs の各要素は normalizeEntry が受け付ける形（単音/和音/休符）。
// opts: {
//   solfege:bool, brackets:[{from,to,text}],   … brackets は単段のときのみ
//   keySig:"G"|"Bb"…  … 調号（フラット/シャープ）を先頭に表示。臨時記号は自動最適化。
//   timeSig:"4/4"      … 拍子記号を先頭に表示。
//   barBeats:4         … この拍数ごとに小節線を引く。
//   measuresPerRow:n   … 段（システム）あたりの小節数の上限。
//   fitWidth:px        … コンテナ幅。密な小節で縮小してつぶれないよう、
//                        この幅に収まる小節数まで自動で段を分ける。
//   bass:[…]           … 指定すると下段（ヘ音記号）に伴奏を描き、大譜表にする。
//                        各要素は normalizeEntry 互換で、小節ごとに barBeats 拍ぶん。
// 8分・16分など旗のある音符は小節内で自動的に連桁（ビーム）でつながる。
// }
// 返り値 host には以下が付く（再生追従用）:
//   host._marks        : 上段エントリごとのハイライト枠（休符は null）
//   host._bassMarks    : 下段（伴奏）エントリごとのハイライト枠（休符は null）
//   host._measureRow   : 小節index → 段index
//   host._measureMark  : 小節index → その小節の先頭音符の枠要素（スクロール位置合わせ用）
//   host._rowCount     : 段数
//   host.setPlayCursor(mi, frac) : 小節 mi 内の進捗 frac(0..1) の位置に再生カーソル（縦線）を表示。
//                                  音符の実座標にそって区分線形補間するので詰まった小節でも自然に動く。
function renderStaff(noteSpecs, opts) {
  opts = opts || {};
  const host = document.createElement("div");
  host.className = "staff-host";
  host._marks = [];
  host._bassMarks = [];
  host._measureRow = [];
  host._measureMark = [];
  host._rowAnchor = [];
  host._rowCount = 1;
  host._measureX = [];         // 小節index → { x1, x2, anchors:[{t,x}] }（カーソル位置の補間用）
  host._rowY = [];             // 段index → { top, bot }（カーソルの縦の範囲）
  host._cursor = null;
  host.setPlayCursor = (mi, frac) => {
    const c = host._cursor;
    if (!c) return;
    const mx = mi != null && mi >= 0 ? host._measureX[mi] : null;
    const ry = mx ? host._rowY[host._measureRow[mi]] : null;
    if (!mx || !ry) { c.style.opacity = 0; return; }
    const f = Math.max(0, Math.min(1, frac || 0));
    let x = mx.x1 + (mx.x2 - mx.x1) * f;                    // 音符が無い小節は線形
    const pts = mx.anchors;
    for (let i = 0; i + 1 < pts.length; i++) {
      if (f >= pts[i].t && f <= pts[i + 1].t) {
        const d = pts[i + 1].t - pts[i].t;
        x = pts[i].x + (d > 0 ? (f - pts[i].t) / d : 0) * (pts[i + 1].x - pts[i].x);
        break;
      }
    }
    c.setAttribute("x1", x); c.setAttribute("x2", x);
    c.setAttribute("y1", ry.top - 8); c.setAttribute("y2", ry.bot + 8);
    c.style.opacity = 1;
  };
  const VF = window.Vex && Vex.Flow;
  if (!VF) { host.textContent = "(楽譜ライブラリを読み込み中…)"; return host; }

  const entries = noteSpecs.map(normalizeEntry);
  entries.forEach((e, i) => { e._gi = i; });
  host._marks = new Array(entries.length).fill(null);

  const hasBrackets = !!(opts.brackets && opts.brackets.length);
  const hasLabels = !!opts.solfege || entries.some((e) => e.label !== undefined);

  // 伴奏（下段・ヘ音記号）。opts.bass は normalizeEntry 互換の配列（小節ごとに barBeats 拍）。
  const hasBass = !!(opts.bass && opts.bass.length);
  let bassEntries = [], bassMeasures = [], bassAbove = 0, bassBelowExtent = 1, bassBelowPx = 0;
  if (hasBass) {
    bassEntries = opts.bass.map(normalizeEntry);
    bassEntries.forEach((e, i) => { e._gi = i; });
    host._bassMarks = new Array(bassEntries.length).fill(null);
    bassMeasures = splitMeasures(bassEntries, opts.barBeats);
    const bDi = bassEntries.flatMap((e) => e.names.map(diOf));
    const bMax = Math.max(26, ...bDi);   // ヘ音記号 上線 A3=26
    const bMin = Math.min(18, ...bDi);   // ヘ音記号 下線 G2=18
    bassAbove = Math.max(0, Math.ceil((bMax - 26) / 2));
    bassBelowExtent = Math.max(1, Math.ceil((18 - bMin) / 2));
    bassBelowPx = Math.max(0, 18 - bMin) * 5;
  }

  // 音域から、五線の上下にはみ出す量を求めて余白・高さを決める（見切れ防止）
  // di = 全音階インデックス。F5(上線)=38、E4(下線)=30。1行=di 2つ分。
  const allDi = entries.flatMap((e) => e.names.map(diOf));
  const diMax = Math.max(38, ...allDi);
  const diMin = Math.min(30, ...allDi);
  const aboveExtent = Math.ceil((diMax - 38) / 2);      // 上線より上の行数
  const belowExtent = Math.max(1, Math.ceil((30 - diMin) / 2)); // 下線より下の行数
  const belowPx = Math.max(0, 30 - diMin) * 5;          // 最低音が下線より下にはみ出すpx
  // 大譜表のときはコード名ラベルを上段の上に出すため、上余白を少し広げる。
  const aboveLn = Math.max(hasBrackets ? 3 : 2, aboveExtent + 2) + (hasBass && hasLabels ? 2 : 0);

  // 小節に分割
  const measures = splitMeasures(entries, opts.barBeats);
  const sigStart = 40 + (opts.keySig ? 10 + (KEYSIG_COUNT[opts.keySig] || 0) * 9 : 0);
  const timeW = opts.timeSig ? 26 : 0;

  // 小節ごとに必要な幅を、その小節自身の音符数（上段・下段の多い方）で見積もる。
  // 密な小節ほど広く要る。全体の最大でそろえると疎な小節が間延びするため個別に。
  const measW = measures.map((m, i) => {
    const n = Math.max(m.length, hasBass ? ((bassMeasures[i] || []).length) : 1);
    return Math.max(96, 22 + n * 44);
  });

  // 段（システム）への割り付け：fitWidth があれば「その幅に収まるだけ」貪欲に詰める。
  // これで疎な所は多く、密な所は少なく自動配分され、縮小によるつぶれを防ぐ。
  const capPerRow = opts.measuresPerRow ? Math.max(1, opts.measuresPerRow) : Infinity;
  const rowsIdx = [];                                   // 各段の小節indexの配列
  if (opts.fitWidth) {
    const budget = Math.max(120, opts.fitWidth - sigStart - timeW - 16);
    let cur = [], curW = 0;
    for (let i = 0; i < measures.length; i++) {
      if (cur.length && (curW + measW[i] > budget || cur.length >= capPerRow)) { rowsIdx.push(cur); cur = []; curW = 0; }
      cur.push(i); curW += measW[i];
    }
    if (cur.length) rowsIdx.push(cur);
  } else {
    const mpr = capPerRow === Infinity ? measures.length : capPerRow;
    for (let i = 0; i < measures.length; i += mpr) rowsIdx.push(measures.map((_, k) => k).slice(i, i + mpr));
  }
  const rows = rowsIdx.map((idxs) => idxs.map((i) => measures[i]));
  const multi = rows.length > 1;
  host._rowCount = rows.length;
  rowsIdx.forEach((idxs, ri) => idxs.forEach((mi) => { host._measureRow[mi] = ri; }));

  // 横幅：全段を同じ幅にそろえて左右をそろえる（実際の楽譜と同様に各段を justify）。
  // fitWidth があればそれに合わせ、無ければ最も内容の多い段に合わせる。
  const rowNatW = rowsIdx.map((idxs, ri) =>
    sigStart + (ri === 0 ? timeW : 0) + idxs.reduce((s, i) => s + measW[i], 0) + 16);
  const naturalMax = Math.max(0, ...rowNatW);
  const width = opts.fitWidth ? Math.max(opts.fitWidth, naturalMax) : naturalMax;

  // 縦：1段の高さ。多段はラベル無しで詰める。大譜表のときは下段のぶんを加える。
  const labelSpace = (hasLabels && !hasBass) ? 42 : 0;
  const braceGap = hasBass ? 34 : 0;                          // 上段下線 と 下段上線 の間隔
  const bassBlock = hasBass ? (bassAbove * 10 + 40 + bassBelowPx) : 0;
  const rowSpan = aboveLn * 10 + 40 + belowPx + braceGap + bassBlock + labelSpace + (multi ? 16 : 8);
  const topPad = 8, botPad = 6;
  const height = rows.length * rowSpan + topPad + botPad + 8;

  try {
    const renderer = new VF.Renderer(host, VF.Renderer.Backends.SVG);
    renderer.resize(width, height);
    const ctx = renderer.getContext();
    const svg = host.querySelector("svg");

    const centerX = (sn) => {
      const a = typeof sn.getNoteHeadBeginX === "function" ? sn.getNoteHeadBeginX() : sn.getAbsoluteX();
      const b = typeof sn.getNoteHeadEndX === "function" ? sn.getNoteHeadEndX() : a + 11;
      return (a + b) / 2;
    };

    let contentTop = Infinity, contentBot = -Infinity, firstTopLineY = 0;
    const allStaveNotes = [];      // {sn, e}
    let measIndex = 0;

    rows.forEach((row, ri) => {
      const staveY = topPad + ri * rowSpan;
      // 段の幅：内容が全幅の7割以上あれば全幅に伸ばして右端をそろえる（通常の段組み）。
      // 内容の少ない段（あぶれた1小節や最終段など）は自然な幅のままにして、
      // 数個の音符が横幅いっぱいに間延びするのを防ぐ。
      const rowW = rowNatW[ri] >= width * 0.7 ? width : rowNatW[ri];
      const stave = new VF.Stave(0, staveY, rowW - 2, { space_above_staff_ln: aboveLn, space_below_staff_ln: hasBass ? 1 : belowExtent + 1 });
      stave.addClef("treble");
      if (opts.keySig) stave.addKeySignature(opts.keySig);
      if (opts.timeSig && ri === 0) stave.addTimeSignature(opts.timeSig);   // 拍子は先頭段のみ
      stave.setContext(ctx);

      // 下段（ヘ音記号）。上段の下線から braceGap ぶん空けて配置。
      let bassStave = null;
      if (hasBass) {
        const bassTopY = stave.getYForLine(4) + belowPx + braceGap;
        bassStave = new VF.Stave(0, bassTopY, rowW - 2, { space_above_staff_ln: 1, space_below_staff_ln: bassBelowExtent + 1 });
        bassStave.addClef("bass");
        if (opts.keySig) bassStave.addKeySignature(opts.keySig);
        if (opts.timeSig && ri === 0) bassStave.addTimeSignature(opts.timeSig);
        bassStave.setContext(ctx);
        // 上下段の音符開始位置を揃える（ト音記号とヘ音記号で幅が違うため）。
        // これで同じ拍の音符・小節線が両段でぴったり縦にそろう。
        const nsx = Math.max(stave.getNoteStartX(), bassStave.getNoteStartX());
        if (typeof stave.setNoteStartX === "function") stave.setNoteStartX(nsx);
        if (typeof bassStave.setNoteStartX === "function") bassStave.setNoteStartX(nsx);
      }

      // 段内の音符＋連桁グループ。小節境界に BarNote を入れて小節を区切る。
      // 各小節は tileMeasure で「正確に表せる音価」に整えるので、上段・下段の
      // レンダー tick が必ず barBeats に一致し、音符・小節線が両段で揃う。
      const tickables = [];
      const rowNotes = [];         // {sn, e, measure} … 元エントリ先頭ピースのみ（ハイライト用）
      const beamGroups = [];       // 連桁対象（小節ごとの StaveNote 配列）
      const barNotes = [];         // 内部小節線の BarNote（連結線の位置決めに使う）
      const ties = [];             // [firstSN, lastSN] … 分割音符をつなぐタイ
      const buildMeasure = (measure, clef, tickList, rowList, groupList, tieList) => {
        const grp = [];
        let tiePrev = null;
        let pos = 0;                                       // 小節内の拍位置（連桁のグループ分け用）
        tileMeasure(measure, opts.barBeats).forEach((p) => {
          const pe = { names: p.base.names, rest: p.base.rest, beats: p.beats };
          const sn = makeStaveNote(pe, opts, VF, clef);
          tickList.push(sn);
          grp.push({ sn, beats: p.beats, rest: !!p.base.rest, pos });
          pos += p.beats;
          if (!p.tieStart && rowList) rowList.push({ sn, e: p.base, measure: measIndex });  // measure は下で上書き
          if (!p.tieStart && clef === "treble") allStaveNotes.push({ sn, e: p.base });
          if (p.tieStart && tiePrev && !p.base.rest) tieList.push([tiePrev, sn]);
          tiePrev = sn;
        });
        groupList.push(grp);
        return grp;
      };
      row.forEach((measure, mi) => {
        if (mi > 0) { const bn = new VF.BarNote(); tickables.push(bn); barNotes.push(bn); }
        const before = rowNotes.length;
        buildMeasure(measure, "treble", tickables, rowNotes, beamGroups, ties);
        for (let k = before; k < rowNotes.length; k++) rowNotes[k].measure = measIndex + mi;
      });

      // 下段（伴奏）の音符＋連桁グループ。上段と同じ位置に BarNote を入れて揃える。
      const bassTickables = [];
      const bassRowNotes = [];
      const bassBeamGroups = [];
      const bassTies = [];
      if (hasBass) {
        row.forEach((measure, mi) => {
          if (mi > 0) bassTickables.push(new VF.BarNote());
          buildMeasure(bassMeasures[measIndex + mi] || [], "bass", bassTickables, bassRowNotes, bassBeamGroups, bassTies);
        });
      }

      const rowBeats = row.reduce((s, m) => s + m.reduce((t, e) => t + (e.beats == null ? 4 : e.beats), 0), 0);
      const nb = Math.max(1, Math.round(rowBeats));
      const voice = new VF.Voice({ num_beats: nb, beat_value: 4 }).setMode(VF.Voice.Mode.SOFT);
      voice.addTickables(tickables);
      voice.setStave(stave);                              // ← 各声部に譜表を割当てて上下段を揃える
      if (opts.keySig) VF.Accidental.applyAccidentals([voice], opts.keySig);
      let bassVoice = null;
      if (hasBass) {
        bassVoice = new VF.Voice({ num_beats: nb, beat_value: 4 }).setMode(VF.Voice.Mode.SOFT);
        bassVoice.addTickables(bassTickables);
        bassVoice.setStave(bassStave);
        if (opts.keySig) VF.Accidental.applyAccidentals([bassVoice], opts.keySig);
      }
      // 音符開始位置は上段・下段のうち広い方（音部記号の幅差を吸収）に合わせる。
      const startX = Math.max(stave.getNoteStartX(), bassStave ? bassStave.getNoteStartX() : 0);
      const fmt = new VF.Formatter();
      fmt.joinVoices([voice]);
      if (bassVoice) fmt.joinVoices([bassVoice]);
      fmt.format(bassVoice ? [voice, bassVoice] : [voice], Math.max(60, rowW - startX - 14));

      // 連桁：小節内の拍位置を自前で数え、同じ半小節（2拍）窓の中で連続する
      // 旗つき音符（8分・付点8分・16分）をつなぐ。休符・4分以上の音符・窓境界で切る。
      // VexFlow の generateBeams は境界を「先頭からの累積」で判定するため、
      // シンコペーションがあると境界がずれて旗が残ることがあり、使わない。
      const beams = [];
      const addBeams = (groups) => groups.forEach((grp) => {
        let run = [], runWin = -1;
        const flush = () => { if (run.length >= 2) beams.push(new VF.Beam(run, true)); run = []; };
        grp.forEach(({ sn, beats, rest, pos }) => {
          if (rest || beats >= 1) { flush(); return; }
          const win = Math.floor(pos / 2 + 1e-6);
          if (run.length && win !== runWin) flush();
          run.push(sn); runWin = win;
        });
        flush();
      });
      addBeams(beamGroups);        // 上段
      addBeams(bassBeamGroups);    // 下段（伴奏）も連桁

      // 最終段の右端は終止線（細＋太）。大譜表は連結記号側で両段を貫く終止線にする。
      const lastRow = ri === rows.length - 1;
      if (lastRow && !hasBass && VF.Barline && VF.Barline.type) stave.setEndBarType(VF.Barline.type.END);

      // 描画：五線 → 連結記号 → 声部 → 連桁
      stave.draw();
      if (bassStave) {
        bassStave.draw();
        const rightType = lastRow && VF.StaveConnector.type.BOLD_DOUBLE_RIGHT != null
          ? VF.StaveConnector.type.BOLD_DOUBLE_RIGHT : VF.StaveConnector.type.SINGLE_RIGHT;
        new VF.StaveConnector(stave, bassStave).setType(VF.StaveConnector.type.BRACE).setContext(ctx).draw();
        new VF.StaveConnector(stave, bassStave).setType(VF.StaveConnector.type.SINGLE_LEFT).setContext(ctx).draw();
        new VF.StaveConnector(stave, bassStave).setType(rightType).setContext(ctx).draw();
      }
      voice.draw(ctx, stave);
      if (bassVoice) bassVoice.draw(ctx, bassStave);
      beams.forEach((bm) => bm.setContext(ctx).draw());
      // タイ（分割した音符をつなぐ）
      const drawTie = (a, b) => { try { new VF.StaveTie({ first_note: a, last_note: b }).setContext(ctx).draw(); } catch (e) {} };
      ties.forEach(([a, b]) => drawTie(a, b));
      bassTies.forEach(([a, b]) => drawTie(a, b));

      // 大譜表：BarNote が描く上段・下段の小節線の隙間（両段の間）を縦線でつなぎ、
      // 一本の貫通した小節線に見せる。位置は BarNote に合わせるので音符とも揃う。
      if (bassStave && svg) {
        const yGapTop = stave.getYForLine(4);       // 上段の下線
        const yGapBot = bassStave.getYForLine(0);   // 下段の上線
        barNotes.forEach((bn) => {
          const x = bn.getAbsoluteX();
          svg.appendChild(mkSvg("line", { x1: x, y1: yGapTop, x2: x, y2: yGapBot, class: "staff-barline" }));
        });
      }

      const topLineY = stave.getYForLine(0);
      const botLineY = stave.getYForLine(4);
      if (ri === 0) firstTopLineY = topLineY;
      const yOf = (di) => botLineY - (di - 30) * 5;
      const sysBot = bassStave ? bassStave.getYForLine(4) : botLineY;
      host._rowY[ri] = { top: topLineY, bot: sysBot };

      // 小節番号（各段の頭。1小節目は省略＝一般的な楽譜の慣習）
      if (svg && multi && ri > 0) {
        const t = mkSvg("text", { x: 3, y: topLineY - 6, class: "staff-measure-num" });
        t.textContent = String(measIndex + 1);
        svg.appendChild(t);
        contentTop = Math.min(contentTop, topLineY - 18);
      }

      // 再生カーソル用の小節座標。x の区切りは
      // 音符開始位置 → 内部小節線(BarNote) → 譜表右端。
      if (svg) {
        const xBounds = [startX, ...barNotes.map((bn) => bn.getAbsoluteX()), stave.getX() + stave.getWidth() - 3];
        row.forEach((measure, k) => {
          const x1 = xBounds[k], x2 = xBounds[k + 1];
          if (!(x2 > x1)) return;
          // カーソル補間アンカー：小節内の各音符（休符含む）の拍位置と中心x
          const grp = beamGroups[k] || [];
          const total = grp.length ? grp[grp.length - 1].pos + grp[grp.length - 1].beats : 0;
          const anchors = total > 0
            ? grp.map(({ sn, pos }) => ({ t: pos / total, x: centerX(sn) })) : [];
          anchors.push({ t: 1, x: x2 - 2 });
          host._measureX[measIndex + k] = { x1, x2, anchors };
        });
      }

      // 段の上端に不可視アンカー（スクロール位置合わせ用。音高に依存しない基準）
      if (svg) {
        const anchor = mkSvg("rect", { x: 0, y: staveY, width: 1, height: 1, fill: "none", stroke: "none" });
        svg.appendChild(anchor);
        host._rowAnchor[ri] = anchor;
      }

      // この段の描画範囲を集計
      contentTop = Math.min(contentTop, topLineY);
      contentBot = Math.max(contentBot, botLineY);
      rowNotes.forEach(({ sn }) => {
        const bb = sn.getBoundingBox && sn.getBoundingBox();
        if (bb) { contentTop = Math.min(contentTop, bb.getY()); contentBot = Math.max(contentBot, bb.getY() + bb.getH()); }
      });
      if (bassStave) {
        contentBot = Math.max(contentBot, bassStave.getYForLine(4) + bassBelowPx + 6);
        bassRowNotes.forEach(({ sn }) => {
          const bb = sn.getBoundingBox && sn.getBoundingBox();
          if (bb) contentBot = Math.max(contentBot, bb.getY() + bb.getH());
        });
      }

      // ラベル（単音のドレミ or コード名）。大譜表では上段の上にコード名を出す。
      if (svg && hasLabels) {
        const labelY = hasBass ? (topLineY - 14) : (botLineY + belowPx + 22);
        rowNotes.forEach(({ sn, e }) => {
          let lbl = e.label;
          const isSolfege = lbl === undefined && opts.solfege && !e.rest && e.names.length === 1;
          if (isSolfege) { const p = parseNote(e.names[0]); lbl = SOLFEGE[p.letter] + accMark(p.acc); }
          if (lbl) {
            const t = mkSvg("text", { x: centerX(sn), y: labelY, class: "staff-label" + (isSolfege ? " solfege" : "") });
            t.textContent = lbl;
            svg.appendChild(t);
          }
        });
        contentTop = hasBass ? Math.min(contentTop, labelY - 12) : contentTop;
        contentBot = hasBass ? contentBot : Math.max(contentBot, labelY + 6);
      }

      // 演奏表現の記譜: 強弱記号(dyn: p/mp/mf/f)・発想標語(mark: rit.等)・スラー(slurNext)
      if (svg) {
        // 強弱は大譜表なら上下段の間、単段なら五線（とラベル）の下
        const dynY = bassStave ? (stave.getYForLine(4) + belowPx + braceGap - 8)
                               : (botLineY + belowPx + (hasLabels ? 40 : 18));
        const markY = topLineY - (hasBass && hasLabels ? 30 : 12);
        rowNotes.forEach(({ sn, e }, ni) => {
          if (e.dyn) {
            const t = mkSvg("text", { x: centerX(sn) - 10, y: dynY, class: "staff-dyn" });
            t.textContent = e.dyn;
            svg.appendChild(t);
            contentBot = Math.max(contentBot, dynY + 6);
          }
          if (e.mark) {
            const t = mkSvg("text", { x: centerX(sn) - 4, y: markY, class: "staff-expr" });
            t.textContent = e.mark;
            svg.appendChild(t);
            contentTop = Math.min(contentTop, markY - 12);
          }
          // スラー: レガートの連続区間(3音以上)を1本の弧で描く。
          // ペアごとに弧を描くと譜面が弧だらけになるため、run 先頭でのみまとめて描画。
          const prevSlur = ni > 0 && rowNotes[ni - 1].e.slurNext;
          if (e.slurNext && !prevSlur && !e.rest) {
            let j = ni;
            while (rowNotes[j + 1] && rowNotes[j].e.slurNext && !rowNotes[j + 1].e.rest) j++;
            if (j - ni + 1 >= 3) {                       // 3音以上つながる場合だけスラーに
              const first = rowNotes[ni], last = rowNotes[j];
              const x1 = centerX(first.sn) + 6, x2 = centerX(last.sn) - 2;
              if (x2 > x1 + 12) {
                let yTop = Infinity;
                for (let k = ni; k <= j; k++)
                  if (rowNotes[k].e.names.length) yTop = Math.min(yTop, yOf(Math.max(...rowNotes[k].e.names.map(diOf))));
                const y1 = yTop - 10;
                const yc = y1 - 8 - Math.min(8, (x2 - x1) * 0.03);
                svg.appendChild(mkSvg("path", { d: `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${yc} ${x2} ${y1}`, class: "staff-slur" }));
              }
            }
          }
        });
        // 松葉（クレシェンド < ・デクレシェンド > ）。opts.hairpins = [{fromGi, toGi, type}]。
        // この段に含まれる範囲だけを描く（段をまたぐ場合は自然に分割される）。
        (opts.hairpins || []).forEach((hp) => {
          const inRow = rowNotes.filter(({ e }) => e._gi >= hp.fromGi && e._gi <= hp.toGi);
          if (inRow.length < 2) return;
          const x1 = centerX(inRow[0].sn), x2 = centerX(inRow[inRow.length - 1].sn) - 8;
          if (x2 - x1 < 24) return;
          const y = dynY - 5, h = 4.5;
          const d = hp.type === "cresc"
            ? `M ${x2} ${y - h} L ${x1} ${y} L ${x2} ${y + h}`
            : `M ${x1} ${y - h} L ${x2} ${y} L ${x1} ${y + h}`;
          svg.appendChild(mkSvg("path", { d, class: "staff-hairpin" }));
          contentBot = Math.max(contentBot, y + h + 4);
        });
      }

      // ハイライト枠。休符にも枠を付ける（再生位置の表示・楽譜エディタでの選択用）。
      // 小節先頭の枠は _measureMark に控える。
      if (svg) {
        rowNotes.forEach(({ sn, e, measure }) => {
          let rect;
          if (e.rest || !e.names.length) {
            rect = mkSvg("rect", { x: centerX(sn) - 9, y: yOf(36), width: 18, height: 20, rx: 6, class: "staff-note-hl rest" });
          } else {
            const dis = e.names.map(diOf);
            const yTop = yOf(Math.max(...dis)) - 8;
            const yBot = yOf(Math.min(...dis)) + 8;
            rect = mkSvg("rect", { x: centerX(sn) - 11, y: yTop, width: 22, height: Math.max(14, yBot - yTop), rx: 6, class: "staff-note-hl" });
          }
          svg.appendChild(rect);
          host._marks[e._gi] = rect;
          if (!host._measureMark[measure]) host._measureMark[measure] = rect;
        });
      }

      // 下段（伴奏）のハイライト枠。ヘ音記号の下線 G2=di18 を基準に配置。休符も対象。
      if (svg && bassStave) {
        const bassBotLineY = bassStave.getYForLine(4);
        const yOfBass = (di) => bassBotLineY - (di - 18) * 5;
        bassRowNotes.forEach(({ sn, e }) => {
          let rect;
          if (e.rest || !e.names.length) {
            rect = mkSvg("rect", { x: centerX(sn) - 9, y: yOfBass(24), width: 18, height: 20, rx: 6, class: "staff-note-hl rest" });
          } else {
            const dis = e.names.map(diOf);
            const yTop = yOfBass(Math.max(...dis)) - 8;
            const yBot = yOfBass(Math.min(...dis)) + 8;
            rect = mkSvg("rect", { x: centerX(sn) - 11, y: yTop, width: 22, height: Math.max(14, yBot - yTop), rx: 6, class: "staff-note-hl" });
          }
          svg.appendChild(rect);
          host._bassMarks[e._gi] = rect;
        });
      }

      measIndex += row.length;
    });

    // 再生カーソル（縦線）。host.setPlayCursor(mi, frac) で移動・表示する。
    if (svg) {
      const cur = mkSvg("line", { x1: -10, y1: 0, x2: -10, y2: 0, class: "staff-cursor" });
      cur.style.opacity = 0;
      svg.appendChild(cur);
      host._cursor = cur;
    }

    // 音程ブラケット（単段のときのみ。五線の上）
    if (svg && hasBrackets && !multi) {
      const y = firstTopLineY - 16;
      contentTop = Math.min(contentTop, y - 18);
      opts.brackets.forEach((b) => {
        const x1 = centerX(allStaveNotes[b.from].sn);
        const x2 = centerX(allStaveNotes[b.to].sn);
        svg.appendChild(mkSvg("path", { d: `M ${x1} ${y + 7} L ${x1} ${y} L ${x2} ${y} L ${x2} ${y + 7}`, class: "staff-bracket-line" }));
        const t = mkSvg("text", { x: (x1 + x2) / 2, y: y - 5, class: "staff-bracket-text" });
        t.textContent = b.text;
        svg.appendChild(t);
      });
    }

    // viewBox を内容の上下端に合わせ、上下の見切れを防ぐ。単段(単一五線)のみ少し拡大。
    if (svg && isFinite(contentTop)) {
      const S = (multi || hasBass) ? 1.0 : 1.22;
      const vbY = contentTop - topPad;
      const vbH = (contentBot + botPad) - vbY;
      svg.setAttribute("viewBox", `0 ${vbY} ${width} ${vbH}`);
      svg.setAttribute("width", Math.round(width * S));
      svg.setAttribute("height", Math.round(vbH * S));
    }
  } catch (e) {
    host.textContent = "♪ " + entries.map((e) => (e.rest ? "𝄽" : e.names.join(""))).join("  ");
  }

  return host;
}
