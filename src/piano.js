// piano.js — 一段鍵盤の描画・PCキーマッピング・発音
// 白鍵 = A段、黒鍵 = Q段。オクターブは可変（鍵盤レイアウトは固定で音高だけシフト）。

// 白鍵に割り当てるPCキー（A段、左から順）
const WHITE_ROW = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", ":", "]"];
// 黒鍵に割り当てるPCキー（Q段）。白鍵#k の直後の黒鍵が QROW[k+1] を使う配置。
const BLACK_ROW = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "@", "["];

// 画面に表示する最大オフセット（左端＝選択キーから約2.5オクターブ）。
const DISPLAY_MAX = 29;

// ピッチクラスが黒鍵か
function isBlackPc(pc) {
  return [1, 3, 6, 8, 10].includes(((pc % 12) + 12) % 12);
}

// オクターブの可動範囲
const MIN_OCTAVE = 1;
const MAX_OCTAVE = 7;


// グランドピアノのサンプル(Salamander Grand Piano / Tone.js ホスト)
const SALAMANDER = {
  urls: {
    A0: "A0.mp3", C1: "C1.mp3", "D#1": "Ds1.mp3", "F#1": "Fs1.mp3", A1: "A1.mp3",
    C2: "C2.mp3", "D#2": "Ds2.mp3", "F#2": "Fs2.mp3", A2: "A2.mp3",
    C3: "C3.mp3", "D#3": "Ds3.mp3", "F#3": "Fs3.mp3", A3: "A3.mp3",
    C4: "C4.mp3", "D#4": "Ds4.mp3", "F#4": "Fs4.mp3", A4: "A4.mp3",
    C5: "C5.mp3", "D#5": "Ds5.mp3", "F#5": "Fs5.mp3", A5: "A5.mp3",
    C6: "C6.mp3", "D#6": "Ds6.mp3", "F#6": "Fs6.mp3", A6: "A6.mp3",
    C7: "C7.mp3", "D#7": "Ds7.mp3", "F#7": "Fs7.mp3", A7: "A7.mp3", C8: "C8.mp3",
  },
  release: 1,
  baseUrl: "https://tonejs.github.io/audio/salamander/",
};

// 選べる楽器。make() は Tone の発音ノードを生成する。
// sampled:true のものは音源ファイルのロード待ちが必要。
const INSTRUMENTS = {
  piano: {
    label: "グランドピアノ", sampled: true, volume: -6,
    make: () => new Tone.Sampler(SALAMANDER),
  },
  epiano: {
    label: "エレクトリックピアノ", volume: -10,
    make: () => new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 3, modulationIndex: 10,
      envelope: { attack: 0.01, decay: 0.6, sustain: 0.2, release: 1.2 },
      modulation: { type: "sine" },
      modulationEnvelope: { attack: 0.01, decay: 0.4, sustain: 0, release: 0.5 },
    }),
  },
  organ: {
    label: "オルガン", volume: -16,
    make: () => new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "fatsquare", count: 3, spread: 20 },
      envelope: { attack: 0.02, decay: 0.1, sustain: 0.9, release: 0.3 },
    }),
  },
  strings: {
    label: "ストリングス", volume: -12,
    make: () => new Tone.PolySynth(Tone.AMSynth, {
      harmonicity: 2,
      oscillator: { type: "sawtooth" },
      envelope: { attack: 0.4, decay: 0.3, sustain: 0.8, release: 1.5 },
    }),
  },
  synth: {
    label: "シンセリード", volume: -12,
    make: () => new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sawtooth" },
      envelope: { attack: 0.02, decay: 0.2, sustain: 0.5, release: 0.6 },
    }),
  },
  marimba: {
    label: "マリンバ", volume: -6,
    make: () => new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: { attack: 0.001, decay: 0.4, sustain: 0, release: 0.4 },
    }),
  },
  bell: {
    label: "ベル", volume: -12,
    make: () => new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 5, modulationIndex: 12,
      envelope: { attack: 0.001, decay: 1.2, sustain: 0, release: 1.2 },
      modulationEnvelope: { attack: 0.001, decay: 0.5, sustain: 0, release: 0.5 },
    }),
  },
};

class Piano {
  constructor(rootEl) {
    this.rootEl = rootEl;
    this.octave = 4;               // ベースのオクターブ（C4 = MIDI60）
    this.tonic = 0;                // ベースのド(=Aキー)の音。Cからの半音数(0=C)
    this.keyEls = {};              // offset -> 鍵盤要素
    this.noteLabels = {};          // offset -> 音名ラベル要素
    this.activeOffsets = new Set();// 現在押されているオフセット
    this.node = null;              // 現在の楽器ノード
    this.instrument = "piano";     // 既定はグランドピアノ
    this.started = false;          // AudioContext開始済みか
    this.ready = false;            // 楽器ロード済みか
    this.listeners = [];
    this._build();
    this._refreshNoteLabels();
    this._updateOctaveLabel();
  }

  // ベースのド(Aキー)のMIDI番号（octave4 & tonic0 -> 60=C4）
  baseMidi() {
    return 12 * (this.octave + 1) + this.tonic;
  }

  // オフセット -> 実際のMIDI番号
  offMidi(off) {
    return this.baseMidi() + off;
  }

  // 現在押されている音を実MIDIの配列で返す
  activeMidi() {
    return [...this.activeOffsets].map((o) => this.offMidi(o)).sort((a, b) => a - b);
  }

  onChange(fn) {
    this.listeners.push(fn);
  }

  _notify() {
    const notes = this.activeMidi();
    this.listeners.forEach((fn) => fn(notes));
  }

  // PCキー -> オフセット の対応を返す
  offsetForKey(key) {
    return this.keyMap[key];
  }

  // 各オフセットが黒鍵か（選択キーtonicからの実際のピッチクラスで判定）
  _isBlack(off) {
    return isBlackPc(this.tonic + off);
  }

  // 鍵盤DOMを生成。左端＝選択キー(tonic)。白鍵/黒鍵は実際のピアノ配置に従う。
  // 白鍵にA段、黒鍵にQ段のPCキーを左から順に割り当てる。
  _build() {
    this.keyMap = {};

    // 白鍵の位置を先に決める
    const whiteOffsets = [];
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      if (!this._isBlack(o)) whiteOffsets.push(o);
    }
    const whiteWidth = 100 / whiteOffsets.length;
    const whitePos = {};
    whiteOffsets.forEach((o, i) => { whitePos[o] = i * whiteWidth; });

    // 左から順にPCキーを割り当てる（白鍵=A段, 黒鍵=Q段）
    let wi = 0;
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      if (this._isBlack(o)) {
        const pcKey = BLACK_ROW[wi]; // 白鍵#(wi-1)の直後 → BLACK_ROW[wi]
        if (pcKey) this.keyMap[pcKey] = o;
      } else {
        const pcKey = WHITE_ROW[wi];
        if (pcKey) this.keyMap[pcKey] = o;
        wi++;
      }
    }
    const offToKey = {};
    for (const [k, o] of Object.entries(this.keyMap)) offToKey[o] = k;

    // 白鍵を描画
    for (const o of whiteOffsets) {
      this.rootEl.appendChild(this._makeKey(o, false, whitePos[o], whiteWidth, offToKey[o]));
    }
    // 黒鍵を描画（左隣の白鍵の右肩。左端が黒鍵のときは左端へ）
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      if (!this._isBlack(o)) continue;
      let lw = o - 1;
      while (lw >= 0 && whitePos[lw] === undefined) lw--;
      const left = lw >= 0 ? whitePos[lw] + whiteWidth * 0.66 : 0;
      this.rootEl.appendChild(this._makeKey(o, true, left, whiteWidth * 0.66, offToKey[o]));
    }
  }

  // tonic変更時などに鍵盤を作り直す
  rebuild() {
    this.activeOffsets.clear();
    this.keyEls = {};
    this.noteLabels = {};
    this.rootEl.innerHTML = "";
    this._build();
    this._refreshNoteLabels();
  }

  _makeKey(off, black, leftPct, widthPct, pcKey) {
    const el = document.createElement("div");
    el.className = "key " + (black ? "black" : "white");
    el.style.left = leftPct + "%";
    el.style.width = widthPct + "%";

    // PCキー名（上）と音名（下）の2行ラベル
    const keyLabel = document.createElement("span");
    keyLabel.className = "key-label";
    keyLabel.textContent = pcKey ? pcKey.toUpperCase() : "";
    el.appendChild(keyLabel);

    const noteLabel = document.createElement("span");
    noteLabel.className = "key-note";
    el.appendChild(noteLabel);
    this.noteLabels[off] = noteLabel;

    el.addEventListener("mousedown", (e) => { e.preventDefault(); this.noteOn(off); });
    el.addEventListener("mouseup", () => this.noteOff(off));
    el.addEventListener("mouseleave", () => this.noteOff(off));

    this.keyEls[off] = el;
    return el;
  }

  _refreshNoteLabels() {
    for (const [off, el] of Object.entries(this.noteLabels)) {
      el.textContent = midiToName(this.offMidi(Number(off)));
    }
  }

  _updateOctaveLabel() {
    const label = document.getElementById("oct-label");
    if (label) label.textContent = `オクターブ ${this.octave}（${midiToName(this.baseMidi())}〜）`;
  }

  // オクターブを設定（範囲内にクランプ）
  setOctave(n) {
    const clamped = Math.max(MIN_OCTAVE, Math.min(MAX_OCTAVE, n));
    if (clamped === this.octave) return;
    this.octave = clamped;
    this._refreshNoteLabels();
    this._updateOctaveLabel();
    this._notify();
  }

  shiftOctave(delta) {
    this.setOctave(this.octave + delta);
  }

  // ベースのド(左端)の音＝キー(調)を設定。pc は Cからの半音数(0-11)。
  // 鍵盤を作り直して、選択キーが左端に来るようスライドさせる。
  setTonic(pc) {
    const t = ((pc % 12) + 12) % 12;
    if (t === this.tonic) return;
    this.tonic = t;
    this.rebuild();
    this._updateOctaveLabel();
    this._notify();
  }

  // オクターブと左端音をまとめて設定（鍵盤の表示窓を一括で移動）。変化なしなら何もしない。
  setWindow(octave, tonic) {
    const o = Math.max(MIN_OCTAVE, Math.min(MAX_OCTAVE, octave));
    const t = ((tonic % 12) + 12) % 12;
    if (o === this.octave && t === this.tonic) return;
    this.octave = o;
    this.tonic = t;
    this.rebuild();
    this._updateOctaveLabel();
    this._notify();
  }

  _setStatus(text) {
    const status = document.getElementById("audio-status");
    if (status) status.textContent = text;
  }

  // 最初のユーザー操作時にAudioContextを開始し、既定の楽器をロード
  async ensureAudio() {
    if (this.started) return;
    await Tone.start();
    this.started = true;
    await this.loadInstrument(this.instrument);
  }

  // 楽器を生成して差し替える（サンプル音源はロードを待つ）
  async loadInstrument(name) {
    const def = INSTRUMENTS[name];
    if (!def) return;
    // 古いノードを破棄
    if (this.node) {
      try { this.node.releaseAll && this.node.releaseAll(); } catch (e) {}
      this.node.dispose();
      this.node = null;
    }
    this.ready = false;
    this._setStatus(`🎵 ${def.label} を読み込み中...`);
    const node = def.make().toDestination();
    if (node.volume) node.volume.value = def.volume ?? -8;
    this.node = node;
    if (def.sampled) await Tone.loaded(); // 全バッファのロードを待つ
    this.ready = true;
    this._setStatus(`🔊 ${def.label}`);
  }

  // 楽器を切り替え（必要ならAudioContextも開始）
  async setInstrument(name) {
    this.instrument = name;
    if (!this.started) await this.ensureAudio();
    else await this.loadInstrument(name);
  }

  noteOn(off) {
    if (this.activeOffsets.has(off)) return;
    this.activeOffsets.add(off);
    const el = this.keyEls[off];
    if (el) el.classList.add("pressed");
    if (this.ready) this.node.triggerAttack(midiToName(this.offMidi(off)));
    this._notify();
  }

  noteOff(off) {
    if (!this.activeOffsets.has(off)) return;
    this.activeOffsets.delete(off);
    const el = this.keyEls[off];
    if (el) el.classList.remove("pressed");
    if (this.ready) this.node.triggerRelease(midiToName(this.offMidi(off)));
    this._notify();
  }

  // 指定の実MIDIを試聴。鍵盤はピッチクラスが一致するものを光らせる。
  playPreview(midiNotes, duration = "2n") {
    if (!this.ready) return;
    this.node.triggerAttackRelease(midiNotes.map(midiToName), duration);
    const pcs = new Set(midiNotes.map(pitchClass));
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      if (pcs.has(pitchClass(this.offMidi(o))) && this.keyEls[o]) {
        const el = this.keyEls[o];
        el.classList.add("hint-flash");
        setTimeout(() => el.classList.remove("hint-flash"), 600);
      }
    }
  }

  // 単音/和音を鍵盤ハイライトなしで鳴らす（流れるメロディ用）。velocity(0-1,省略可)で強弱。
  playTones(names, durationSec, velocity) {
    if (this.ready && names) this.node.triggerAttackRelease(names, durationSec, undefined, velocity);
  }

  // ---- リズム編用の合成ドラムキット（選択楽器とは独立）----
  // 遅延生成（AudioContext開始後にトリガーされる）。Tone.js の合成音でドラムを作る。
  _ensureDrums() {
    if (this._drums) return;
    const bus = new Tone.Gain(0.9).toDestination();
    // キック：低い MembraneSynth
    const kick = new Tone.MembraneSynth({ pitchDecay: 0.03, octaves: 5, oscillator: { type: "sine" }, envelope: { attack: 0.001, decay: 0.30, sustain: 0, release: 0.08 } });
    kick.volume.value = 2; kick.connect(bus);
    // スネア：白色ノイズ＋バンドパス
    const snare = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay: 0.16, sustain: 0 } });
    const snareFilt = new Tone.Filter(1700, "bandpass"); snareFilt.Q.value = 0.9;
    snare.connect(snareFilt); snareFilt.connect(bus); snare.volume.value = -5;
    // ハイハット：ごく短い白色ノイズ＋ハイパス
    const hat = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay: 0.03, sustain: 0 } });
    const hatFilt = new Tone.Filter(8000, "highpass"); hat.connect(hatFilt); hatFilt.connect(bus); hat.volume.value = -14;
    // 木魚系（拍のクリック／クラーベ）：短い MembraneSynth
    const wood = new Tone.MembraneSynth({ pitchDecay: 0.008, octaves: 2, oscillator: { type: "triangle" }, envelope: { attack: 0.001, decay: 0.11, sustain: 0, release: 0.02 } });
    wood.volume.value = -5; wood.connect(bus);
    this._drums = { bus, kick, snare, hat, wood };
  }

  // 層の pitch からドラムの種類を推定（curriculum の慣習：C2/3=キック, A=スネア, F#=ハイハット, E=クラーベ, C5/6=クリック）
  drumForPitch(pitch) {
    const p = pitch || "C5";
    if (p[0] === "C" && (p.indexOf("1") >= 0 || p.indexOf("2") >= 0 || p.indexOf("3") >= 0)) return "kick";
    if (p[0] === "A") return "snare";
    if (p.indexOf("F#") === 0 || p.indexOf("G#") === 0) return "hihat";
    if (p[0] === "E") return "clave";
    return "click";
  }

  // ドラムを1発鳴らす。accent=強拍（大きめ・明るめ）。
  playDrum(type, accent) {
    if (!this.started) return;
    this._ensureDrums();
    const d = this._drums;
    try {
      if (type === "kick") d.kick.triggerAttackRelease("C1", "8n", undefined, accent ? 1 : 0.82);
      else if (type === "snare") d.snare.triggerAttackRelease("16n", undefined, accent ? 1 : 0.7);
      else if (type === "hihat") d.hat.triggerAttackRelease("32n", undefined, accent ? 0.95 : 0.6);
      else if (type === "clave") d.wood.triggerAttackRelease(accent ? "C6" : "G5", "16n", undefined, accent ? 1 : 0.8);
      else d.wood.triggerAttackRelease(accent ? "C5" : "C4", "16n", undefined, accent ? 1 : 0.72); // click
    } catch (e) {}
  }

  // ヒント表示（ピッチクラスで一致する鍵盤を縁取り）。オクターブに依存しない。
  setHints(midiNotes) {
    const pcs = new Set(midiNotes.map(pitchClass));
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      const el = this.keyEls[o];
      if (el) el.classList.toggle("hint", pcs.has(pitchClass(this.offMidi(o))));
    }
  }

  // 特定の実MIDI（その積み方＝ボイシング）だけを光らせる。転回形の可視化用。
  setHintsMidi(midiNotes) {
    const offs = new Set(midiNotes.map((m) => m - this.baseMidi()));
    for (let o = 0; o <= DISPLAY_MAX; o++) {
      const el = this.keyEls[o];
      if (el) el.classList.toggle("hint", offs.has(o));
    }
  }
}
