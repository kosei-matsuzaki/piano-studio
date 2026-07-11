// music.js — 音名・コード理論まわりの共通ロジック
// 音楽的な知識（コード定義、構成音の計算、コード判定）をここに集約する。

// 12音の音名（シャープ表記）。インデックスがピッチクラス(0=C 〜 11=B)に対応。
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// コードの種類。intervals はルートからの半音数。degrees は度数表記（表示用）。
const CHORD_TYPES = {
  maj:    { label: "メジャー",          symbol: "",      intervals: [0, 4, 7],      degrees: ["1", "3", "5"] },
  min:    { label: "マイナー",          symbol: "m",     intervals: [0, 3, 7],      degrees: ["1", "♭3", "5"] },
  dim:    { label: "ディミニッシュ",    symbol: "dim",   intervals: [0, 3, 6],      degrees: ["1", "♭3", "♭5"] },
  aug:    { label: "オーギュメント",    symbol: "aug",   intervals: [0, 4, 8],      degrees: ["1", "3", "#5"] },
  sus4:   { label: "サスフォー",        symbol: "sus4",  intervals: [0, 5, 7],      degrees: ["1", "4", "5"] },
  sus2:   { label: "サスツー",          symbol: "sus2",  intervals: [0, 2, 7],      degrees: ["1", "2", "5"] },
  maj7:   { label: "メジャーセブン",    symbol: "maj7",  intervals: [0, 4, 7, 11],  degrees: ["1", "3", "5", "7"] },
  min7:   { label: "マイナーセブン",    symbol: "m7",    intervals: [0, 3, 7, 10],  degrees: ["1", "♭3", "5", "♭7"] },
  dom7:   { label: "ドミナントセブン",  symbol: "7",     intervals: [0, 4, 7, 10],  degrees: ["1", "3", "5", "♭7"] },
  m7b5:   { label: "ハーフディミニッシュ", symbol: "m7♭5", intervals: [0, 3, 6, 10], degrees: ["1", "♭3", "♭5", "♭7"] },
  six:    { label: "シックス",          symbol: "6",     intervals: [0, 4, 7, 9],   degrees: ["1", "3", "5", "6"] },
  m6:     { label: "マイナーシックス",  symbol: "m6",    intervals: [0, 3, 7, 9],   degrees: ["1", "♭3", "5", "6"] },
  mM7:    { label: "マイナーメジャー7", symbol: "mM7",   intervals: [0, 3, 7, 11],  degrees: ["1", "♭3", "5", "7"] },
  add9:   { label: "アドナイン",        symbol: "add9",  intervals: [0, 4, 7, 14],  degrees: ["1", "3", "5", "9"] },
  // テンションコード（9th/11th/13th）
  dom9:   { label: "ナインス",          symbol: "9",     intervals: [0, 4, 7, 10, 14],     degrees: ["1", "3", "5", "♭7", "9"] },
  dom11:  { label: "イレブンス",        symbol: "11",    intervals: [0, 7, 10, 14, 17],    degrees: ["1", "5", "♭7", "9", "11"] },
  dom13:  { label: "サーティーン",      symbol: "13",    intervals: [0, 4, 10, 14, 21],    degrees: ["1", "3", "♭7", "9", "13"] },
  maj9:   { label: "メジャーナイン",    symbol: "maj9",  intervals: [0, 4, 7, 11, 14],     degrees: ["1", "3", "5", "7", "9"] },
  min9:   { label: "マイナーナイン",    symbol: "m9",    intervals: [0, 3, 7, 10, 14],     degrees: ["1", "♭3", "5", "♭7", "9"] },
  min11:  { label: "マイナーイレブン",  symbol: "m11",   intervals: [0, 3, 7, 10, 14, 17], degrees: ["1", "♭3", "5", "♭7", "9", "11"] },
  madd9:  { label: "マイナーアドナイン", symbol: "m(add9)", intervals: [0, 3, 7, 14],       degrees: ["1", "♭3", "5", "9"] },
};

// コードタイプの簡単な解説（コードガイド用）。
const CHORD_DESC = {
  maj:  "明るく安定した響き。最も基本的なコード。",
  min:  "暗く切ない響き。3度が半音下がるだけでメジャーから変わる。",
  dim:  "緊張感のある不安定な響き。経過的に使われることが多い。",
  aug:  "ふわっと浮くような不安定な響き。",
  sus4: "解決を求めるような宙吊りの響き。majへ進みたくなる。",
  sus2: "開放的で澄んだ響き。",
  maj7: "おしゃれで柔らかい響き。",
  min7: "落ち着いたジャジーな響き。",
  dom7: "次へ進みたくなる推進力のある響き。ブルースの基本。",
  m7b5: "ジャズのツーファイブで使われる繊細な響き。",
  six:  "ポップで明るく軽快な響き。",
  add9: "広がりのあるきらびやかな響き。",
};

// ---- コード進行（ダイアトニックコード）まわり ----

// メジャースケールの各度数のルート(主音からの半音数)
const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
// 各度数のダイアトニックコード(トライアド / セブンス)。index 0=I 〜 6=vii
const DIATONIC_TRIAD = ["maj", "min", "min", "maj", "maj", "min", "dim"];
const DIATONIC_SEVENTH = ["maj7", "min7", "min7", "maj7", "dom7", "min7", "m7b5"];
// ローマ数字表記（大文字=メジャー系, 小文字=マイナー系）
const ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii°"];

// よく使われるコード進行。degrees は度数(0始まり)の並び。
const PROGRESSIONS = {
  ohdo:    { label: "王道進行 (I–V–vi–IV)",   degrees: [0, 4, 5, 3],          seventh: false, desc: "J-POPで最頻出。明るさと切なさが同居する王道の響き。" },
  komuro:  { label: "小室進行 (vi–IV–V–I)",   degrees: [5, 3, 4, 0],          seventh: false, desc: "マイナーで始まり高揚していく。90年代J-POPの定番。" },
  canon:   { label: "カノン進行",             degrees: [0, 4, 5, 2, 3, 0, 3, 4], seventh: false, desc: "パッヘルベルのカノン。壮大で感動的に展開する。" },
  fifties: { label: "50s進行 (I–vi–IV–V)",   degrees: [0, 5, 3, 4],          seventh: false, desc: "オールディーズの定番。優しく懐かしい雰囲気。" },
  twofive: { label: "ツーファイブワン (ii–V–I)", degrees: [1, 4, 0],          seventh: true,  desc: "ジャズの最重要進行。強い解決感が得られる。" },
  basic:   { label: "基本進行 (I–IV–V–I)",    degrees: [0, 3, 4, 0],          seventh: false, desc: "最も基本的なスリーコード。多くの曲の土台。" },
};

// キー(主音名)と進行キーから、各コードの情報を計算する。
// 返り値: [{ roman, rootName, typeKey, chordName, degree }]
function buildProgression(keyRoot, progKey, useSeventh) {
  const prog = PROGRESSIONS[progKey];
  const seventh = useSeventh ?? prog.seventh;
  const tonicPc = NOTE_NAMES.indexOf(keyRoot);
  const table = seventh ? DIATONIC_SEVENTH : DIATONIC_TRIAD;
  return prog.degrees.map((d) => {
    const rootPc = (tonicPc + MAJOR_SCALE[d]) % 12;
    const rootName = NOTE_NAMES[rootPc];
    const typeKey = table[d];
    let roman = ROMAN[d];
    if (seventh) roman += (typeKey === "dom7" ? "7" : typeKey === "m7b5" ? "ø7" : "7");
    return { roman, rootName, typeKey, chordName: makeChordName(rootName, typeKey), degree: d };
  });
}

// コード文字列を {root, type} に解析する。例: "Am7" -> {root:"A", type:"min7"}
const FLAT_TO_SHARP = { Db: "C#", Eb: "D#", Gb: "F#", Ab: "G#", Bb: "A#" };
const SUFFIX_TO_TYPE = {
  "": "maj", "M": "maj", "maj": "maj",
  "m": "min", "min": "min", "-": "min",
  "7": "dom7", "dom7": "dom7",
  "maj7": "maj7", "M7": "maj7",
  "m7": "min7", "min7": "min7", "-7": "min7",
  "dim": "dim", "dim7": "dim",
  "aug": "aug", "+": "aug",
  "sus4": "sus4", "sus": "sus4", "sus2": "sus2",
  "6": "six", "add9": "add9", "m7b5": "m7b5",
};
function parseChord(str) {
  const m = String(str).match(/^([A-G])([#b]?)(.*)$/);
  if (!m) return { root: "C", type: "maj" };
  let root = m[1] + m[2];
  if (FLAT_TO_SHARP[root]) root = FLAT_TO_SHARP[root];
  const type = SUFFIX_TO_TYPE[m[3]] ?? "maj";
  return { root, type };
}

// キー(主音名)の中でのローマ数字表記を返す（ダイアトニック外は空文字）
function romanInKey(keyRoot, rootName, typeKey) {
  const tonicPc = NOTE_NAMES.indexOf(keyRoot);
  const pc = NOTE_NAMES.indexOf(rootName);
  const deg = MAJOR_SCALE.indexOf(((pc - tonicPc) % 12 + 12) % 12);
  if (deg === -1) return "";
  let r = ROMAN[deg];
  if (typeKey.includes("7")) r += "7";
  return r;
}

// MIDIノート番号 → 音名（オクターブ付き）。例: 60 -> "C4"
function midiToName(midi) {
  const name = NOTE_NAMES[midi % 12];
  const octave = Math.floor(midi / 12) - 1;
  return name + octave;
}

// MIDIノート番号 → ピッチクラス(0-11)
function pitchClass(midi) {
  return ((midi % 12) + 12) % 12;
}

// 音名(オクターブ付き、♯/♭可) → MIDIノート番号。例: "C4" -> 60, "Eb4" -> 63
function noteNameToMidi(name) {
  const m = String(name).match(/^([A-G])([#b]?)(-?\d+)$/);
  if (!m) return 60;
  const shift = m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0;
  return NOTE_NAMES.indexOf(m[1]) + (parseInt(m[3], 10) + 1) * 12 + shift;
}

// ルート音名 + コードタイプ → MIDIノート番号の配列
// rootMidi はルートのMIDI番号（例: C4 = 60）
function chordNotes(rootMidi, typeKey) {
  const type = CHORD_TYPES[typeKey];
  return type.intervals.map((i) => rootMidi + i);
}

// コード名（表示用）。例: makeChordName("C", "maj7") -> "Cmaj7"
function makeChordName(rootName, typeKey) {
  return rootName + CHORD_TYPES[typeKey].symbol;
}

// コードを「正しい綴り」の音名配列にする（楽譜の和音表示用）。
//   例: spellChord("C","min",4) -> ["C4","Eb4","G4"]（D#ではなくEb）
//       spellChord("F","min",4) -> ["F4","Ab4","C5"]、spellChord("G","aug",4) -> ["G4","B4","D#4"]
// 度数(degrees)に基づいて音名の文字を決め、臨時記号は音高に合わせる。
const LETTERS = ["C", "D", "E", "F", "G", "A", "B"];
const LETTER_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function spellChord(root, type, baseOct) {
  baseOct = baseOct == null ? 4 : baseOct;
  const m = String(root).match(/^([A-G])(##|bb|#|b|)$/);
  if (!m) return [root + baseOct];
  const rootLetterIdx = LETTERS.indexOf(m[1]);
  const rootShift = m[2] === "#" ? 1 : m[2] === "##" ? 2 : m[2] === "b" ? -1 : m[2] === "bb" ? -2 : 0;
  const rootPc = ((LETTER_PC[m[1]] + rootShift) % 12 + 12) % 12;
  const t = CHORD_TYPES[type];
  if (!t) return [root + baseOct];

  return t.intervals.map((iv, idx) => {
    const degNum = parseInt(String(t.degrees[idx]).replace(/[^0-9]/g, ""), 10) || 1;
    const step = degNum - 1;                              // ルート文字からの音度ステップ
    const letterIdx = (rootLetterIdx + step) % 7;
    const octShift = Math.floor((rootLetterIdx + step) / 7);
    const targetPc = (rootPc + iv) % 12;
    let acc = ((targetPc - LETTER_PC[LETTERS[letterIdx]]) % 12 + 12) % 12;
    if (acc > 6) acc -= 12;                               // -5..6 に正規化
    const accStr = acc === 1 ? "#" : acc === 2 ? "##" : acc === -1 ? "b" : acc === -2 ? "bb" : "";
    return LETTERS[letterIdx] + accStr + (baseOct + octShift);
  });
}

// 転回形の名前
const INVERSION_NAMES = ["基本形", "第1転回形", "第2転回形", "第3転回形"];

// コードの転回形をすべて返す（[0]=基本形, [1]=第1転回形, ...）。
// 最低音を1オクターブ上へ移すことで転回していく。
function chordInversions(rootMidi, typeKey) {
  const base = chordNotes(rootMidi, typeKey);
  const result = [base.slice()];
  let cur = base.slice();
  for (let i = 1; i < base.length; i++) {
    cur = cur.slice();
    cur.push(cur.shift() + 12);
    result.push(cur);
  }
  return result;
}

// 押されている音(ピッチクラスの集合)から該当するコードを推定する。
// notes は MIDI番号の配列。一致するものがなければ null。
// 転回形も拾えるよう、すべてのルート候補で照合する。
function detectChord(midiNotes) {
  const classes = [...new Set(midiNotes.map(pitchClass))].sort((a, b) => a - b);
  if (classes.length < 3) return null;

  for (const root of classes) {
    // ルートを基準にした半音差の集合を作る
    const relative = new Set(classes.map((c) => ((c - root + 12) % 12)));
    for (const [key, type] of Object.entries(CHORD_TYPES)) {
      const target = new Set(type.intervals.map((i) => i % 12));
      if (target.size === relative.size && [...target].every((t) => relative.has(t))) {
        return makeChordName(NOTE_NAMES[root], key);
      }
    }
  }
  return null;
}

// 与えられた音(MIDI)が、目標コードの構成音(ピッチクラス)を過不足なく満たすか判定。
// オクターブは無視する。クイズの正誤判定に使う。
function matchesChord(midiNotes, rootMidi, typeKey) {
  const want = new Set(chordNotes(rootMidi, typeKey).map(pitchClass));
  const got = new Set(midiNotes.map(pitchClass));
  if (want.size !== got.size) return false;
  return [...want].every((p) => got.has(p));
}
