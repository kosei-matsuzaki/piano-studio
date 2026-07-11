// neural.js — 学習済みLSTM（model.js）でコード進行とメロディを生成するブラウザ推論エンジン。
// 依存なし（自前の行列演算）。gate順序は学習側(train.py)と揃えて [i, f, g, o]。
//   ・コードモデル   : 直前のコードから次のコードを予測（キー正規化されたローマ数字列）
//   ・メロディモデル : 「直前の音」と「現在のコード」から次の音(音程+音価)を予測
// model.js が読み込まれていない場合は NeuralComposer.available() が false を返す。

(function () {
  "use strict";

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  // W: 行の配列（各行=入力次元） → 出力ベクトル
  function matVec(W, x) {
    const out = new Float64Array(W.length);
    for (let r = 0; r < W.length; r++) {
      const row = W[r]; let s = 0;
      for (let k = 0; k < row.length; k++) s += row[k] * x[k];
      out[r] = s;
    }
    return out;
  }

  // LSTM 1層1ステップ。L={Wih,Whh,b}。z = Wih·x + Whh·h + b をゲート分割。
  function lstmCell(x, h, c, L) {
    const H = h.length;
    const zi = matVec(L.Wih, x), zh = matVec(L.Whh, h), b = L.b;
    const hn = new Float64Array(H), cn = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      const i = sigmoid(zi[j] + zh[j] + b[j]);
      const f = sigmoid(zi[H + j] + zh[H + j] + b[H + j]);
      const g = Math.tanh(zi[2 * H + j] + zh[2 * H + j] + b[2 * H + j]);
      const o = sigmoid(zi[3 * H + j] + zh[3 * H + j] + b[3 * H + j]);
      const cc = f * c[j] + i * g;
      cn[j] = cc; hn[j] = o * Math.tanh(cc);
    }
    return { h: hn, c: cn };
  }

  // 多層LSTMを1ステップ実行。states[l]={h,c} を更新し、最上層の h を返す。
  function runStack(x, states, layers) {
    let inp = x;
    for (let l = 0; l < layers.length; l++) {
      const st = lstmCell(inp, states[l].h, states[l].c, layers[l]);
      states[l] = st; inp = st.h;
    }
    return inp;
  }

  // 各層の状態(h,c)をゼロで初期化。
  function newStates(nlayers, H) {
    const s = [];
    for (let l = 0; l < nlayers; l++) s.push({ h: zeros(H), c: zeros(H) });
    return s;
  }

  // 重みを層配列に正規化（新形式 layers[] / 旧形式 flat Wih,Whh,b の両対応）。
  function toLayers(src) {
    return src.layers ? src.layers : [{ Wih: src.Wih, Whh: src.Whh, b: src.b }];
  }

  function logits(h, m) {
    const y = matVec(m.Wout, h);
    for (let k = 0; k < y.length; k++) y[k] += m.bout[k];
    return y;
  }

  const EMPTY = new Float64Array(0);
  // ベクトルを連結（可変個）
  function concat(...vs) {
    let n = 0; for (const v of vs) n += v.length;
    const out = new Float64Array(n); let o = 0;
    for (const v of vs) { out.set(v, o); o += v.length; }
    return out;
  }
  // ジャンル埋め込みを取得（モデルが非対応 or ジャンル不明なら空/ popフォールバック）
  function genreVec(m, genre) {
    if (!m.embG) return EMPTY;
    let gi = m.gIdx[genre];
    if (gi == null) gi = m.gIdx.pop != null ? m.gIdx.pop : 0;
    return m.embG[gi];
  }

  // ---- Transformer（自己注意）用の演算。全部依存なしの自作。 ----
  function addv(a, b) { const o = new Float64Array(a.length); for (let k = 0; k < a.length; k++) o[k] = a[k] + b[k]; return o; }
  function layerNorm(v, g, b) {                          // eps=1e-5（PyTorch既定）
    const n = v.length; let m = 0; for (let k = 0; k < n; k++) m += v[k]; m /= n;
    let va = 0; for (let k = 0; k < n; k++) { const d = v[k] - m; va += d * d; } va /= n;
    const inv = 1 / Math.sqrt(va + 1e-5);
    const o = new Float64Array(n); for (let k = 0; k < n; k++) o[k] = (v[k] - m) * inv * g[k] + b[k];
    return o;
  }
  function ffn(v, L) {                                    // W2·relu(W1·v+b1)+b2
    const h = matVec(L.W1, v); for (let k = 0; k < h.length; k++) h[k] = Math.max(0, h[k] + L.b1[k]);
    const o = matVec(L.W2, h); for (let k = 0; k < o.length; k++) o[k] += L.b2[k];
    return o;
  }
  // 因果的マルチヘッド自己注意。Xn=[L][d]（LN済み） → [L][d]。
  function mhaCausal(Xn, ly, nHeads, d) {
    const Ln = Xn.length, dh = d / nHeads;
    const Q = Xn.map((v) => addv(matVec(ly.Wq, v), ly.bq));
    const K = Xn.map((v) => addv(matVec(ly.Wk, v), ly.bk));
    const V = Xn.map((v) => addv(matVec(ly.Wv, v), ly.bv));
    const scale = 1 / Math.sqrt(dh);
    const out = [];
    for (let i = 0; i < Ln; i++) {
      const o = new Float64Array(d);
      for (let head = 0; head < nHeads; head++) {
        const off = head * dh;
        let mx = -Infinity; const sc = new Float64Array(i + 1);
        for (let j = 0; j <= i; j++) { let s = 0; for (let t = 0; t < dh; t++) s += Q[i][off + t] * K[j][off + t]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
        let sum = 0; for (let j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
        for (let j = 0; j <= i; j++) { const w = sc[j] / sum; for (let t = 0; t < dh; t++) o[off + t] += w * V[j][off + t]; }
      }
      out.push(addv(matVec(ly.Wo, o), ly.bo));           // 出力射影
    }
    return out;
  }
  // 1つのクエリ q を、キャッシュ済み K/V 全位置に対して注意（KVキャッシュ生成用）。→ [d]
  function attendOneQuery(q, Karr, Varr, nHeads, d) {
    const dh = d / nHeads, P = Karr.length, scale = 1 / Math.sqrt(dh);
    const out = new Float64Array(d);
    for (let head = 0; head < nHeads; head++) {
      const off = head * dh; let mx = -Infinity; const sc = new Float64Array(P);
      for (let j = 0; j < P; j++) { let s = 0; for (let t = 0; t < dh; t++) s += q[off + t] * Karr[j][off + t]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
      let sum = 0; for (let j = 0; j < P; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
      for (let j = 0; j < P; j++) { const w = sc[j] / sum; for (let t = 0; t < dh; t++) out[off + t] += w * Varr[j][off + t]; }
    }
    return out;
  }

  // 温度つきサンプリング。banned は禁止するインデックスの集合。
  function sample(logitArr, temperature, banned) {
    const t = Math.max(0.05, temperature);
    let max = -Infinity;
    for (let k = 0; k < logitArr.length; k++) {
      if (banned && banned.has(k)) continue;
      const v = logitArr[k] / t;
      if (v > max) max = v;
    }
    let sum = 0; const p = new Float64Array(logitArr.length);
    for (let k = 0; k < logitArr.length; k++) {
      if (banned && banned.has(k)) { p[k] = 0; continue; }
      const e = Math.exp(logitArr[k] / t - max); p[k] = e; sum += e;
    }
    let r = Math.random() * sum;
    for (let k = 0; k < logitArr.length; k++) { r -= p[k]; if (r <= 0) return k; }
    // フォールバック（数値誤差時）
    for (let k = 0; k < logitArr.length; k++) if (p[k] > 0) return k;
    return 0;
  }

  function zeros(n) { return new Float64Array(n); }

  // ---- 保持トークン ----
  // 学習コーパスは「1トークン=1拍、"_"=直前コードの保持」の保持トークン形式。
  // 生成後は展開して従来どおり「拍ごとのローマ数字列」に戻す（旧モデル＝"_"なし語彙でも無害）。
  const HOLD = "_";
  function expandHolds(toks) {
    const out = []; let prev = "I";
    for (const t of toks) {
      if (t !== HOLD) prev = t;
      out.push(prev);
    }
    return out;
  }
  function distinctChords(toks) {
    return new Set(toks.filter((t) => t !== HOLD)).size;
  }
  // ローマ数字の基幹（変化記号+度数+大小）。装飾(7th/sus/分数等)を落として比較する用途。
  function romanBaseOf(r) {
    const m = String(r).split("/")[0].match(/^([b#]*)(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)/);
    return m ? m[1] + m[2] : String(r);
  }
  // 拍コード列から短調（vi中心）かを推定。データは平行長調に正規化されているので、
  // 短調の曲は vi をトニックとする進行として現れる。
  function isMinorRomans(seq) {
    let wI = 0, wVi = 0;
    for (const r of seq) { const b = romanBaseOf(r); if (b === "I") wI++; else if (b === "vi") wVi++; }
    return wVi > wI * 1.2;
  }
  // 終止のソフト誘導: 固定のカデンツ付加はしない。かわりに EOS(曲の終わり)のロジットを
  // 「今トニック系(I/vi)のコードにいるか」で増減させ、終わりやすい場所で曲が閉じるようにする。
  // データでもトニック終止は54%（フェードアウト等でV/IV終止も多い）なので、禁止ではなくバイアス。
  const EOS_TONIC_BONUS = 0.5, EOS_OFFTONIC_DROP = 1.25;
  function biasEos(y, eosIdx, curChord) {
    if (eosIdx == null || curChord == null) return y;
    const b = romanBaseOf(curChord);
    y[eosIdx] += (b === "I" || b === "vi") ? EOS_TONIC_BONUS : -EOS_OFFTONIC_DROP;
    return y;
  }
  // maxBars 打ち切り（EOSでなく強制終了）のとき、モデルが作った進行の中の
  // 「最後にトニック(I/vi)で終わる小節」まで戻って曲を閉じる。カデンツの付加はしない。
  function trimToTonic(seq, minBeats) {
    for (let bar = Math.floor(seq.length / 4) - 1; bar * 4 >= minBeats; bar--) {
      const b = romanBaseOf(seq[bar * 4 + 3]);
      if (b === "I" || b === "vi") return seq.slice(0, (bar + 1) * 4);
    }
    return seq;
  }
  // 拍位置で保持のロジットを調整し、コードチェンジを強拍(1・3拍目)へ寄せる。
  // 学習データの変化は87%が強拍だが、自由生成は位置埋め込みより直近パターンに引きずられて
  // 位相がずれたまま進みやすい。弱拍は保持を優遇し、小節頭は保持を軽く抑制して変化率を保つ
  // （弱拍優遇だけだと「長い保持」が自己強化されて進行が静的になりすぎる）。
  // 禁止でなくバイアスなので、食い気味の変化も低頻度では残る。値は churn/位相の実測でチューニング済み。
  const WEAK_BEAT_HOLD_BIAS = 2.5;   // 2・4拍目: 保持を優遇
  const BAR_HEAD_HOLD_DROP = 1.25;   // 1拍目(小節頭): 保持を抑制
  function biasWeakBeat(y, holdIdx, beatIndex) {
    if (holdIdx == null) return y;
    const ph = beatIndex % 4;
    if (ph === 1 || ph === 3) y[holdIdx] += WEAK_BEAT_HOLD_BIAS;
    else if (ph === 0) y[holdIdx] -= BAR_HEAD_HOLD_DROP;
    return y;
  }
  // 同一コードが MAX_RUN_BEATS(2小節)を超えて続いたら、続くほど強いペナルティを
  // 「保持トークン」と「同じコードの再選択」の両方に掛ける（データでは8拍超の保持は1.2%）。
  // 同一コードは8拍(2小節)で実質打ち止め。以前の漸増ペナルティ(-0.5/拍)は弱拍の保持
  // ボーナス(+2.5)と相殺して10〜12拍の保持が抜けていたため、相殺不能な強ペナルティにする。
  const MAX_RUN_BEATS = 8, RUN_PENALTY = 12;
  function biasLongRun(y, holdIdx, sameIdx, runBeats) {
    if (runBeats < MAX_RUN_BEATS) return y;
    if (holdIdx != null) y[holdIdx] -= RUN_PENALTY;
    if (sameIdx != null) y[sameIdx] -= RUN_PENALTY;
    return y;
  }
  // 語彙の偏り対策: Transformerは最初の数コードの特徴(I-IV-Vだけ/maj7だらけ/分数だらけ)に
  // 文脈が自己強化で引っ張られる。曲内の出現率が上限を超えたトークン/カテゴリに超過比例の
  // ペナルティを掛けて分布をデータ水準(maj7 4.5%・sus 3.8%・分数7.4%)に近づける。
  const TOKEN_SHARE_CAP = 0.28;                               // 1コードの出現率上限
  const CAT_SHARE_CAP = { slash: 0.15, maj7: 0.12, sus: 0.10 };
  function catCountsOf(vocab) {
    const idx = { slash: [], maj7: [], sus: [] };
    for (let i = 0; i < vocab.length; i++) {
      const t = vocab[i];
      if (t === HOLD || t[0] === "<") continue;
      if (t.includes("/")) idx.slash.push(i);
      if (/maj7|maj9|mM7/.test(t)) idx.maj7.push(i);
      if (/sus/.test(t)) idx.sus.push(i);
    }
    return idx;
  }
  function biasDiversity(y, idxMap, catIdx, tokCount, catCount, evCount) {
    if (evCount < 6) return y;                                // 曲頭はノイズなので判定しない
    for (const [tok, c] of tokCount) {
      const over = c / evCount - TOKEN_SHARE_CAP;
      if (over > 0) { const ti = idxMap[tok]; if (ti != null) y[ti] -= 40 * over; }
    }
    for (const k in CAT_SHARE_CAP) {
      const over = catCount[k] / evCount - CAT_SHARE_CAP[k];
      if (over > 0) { const pen = 50 * over; for (const i of catIdx[k]) y[i] -= pen; }
    }
    return y;
  }
  // 彩りコード(maj7/sus/aug/dim/6th/add9/テンション/分数)へのロジットボーナス。
  // データでは彩りコードが約11%を占めるが、構造の反復のため chordTemp<1 で生成すると
  // 分布が尖って裾の彩りがほぼ消える。温度を上げる代わりに彩りクラスだけ底上げする
  // （クラス内の相対確率は不変なので、モデルが自然と思う文脈にだけ浮上する）。
  // 一律ボーナスだとモデルが元々出しやすい sus ばかり増幅されるので、カテゴリ別に調整
  // （目標はデータの内訳: maj7系4.5% / sus3.8% / その他2%台）。素の7th(V7等)は対象外。
  const COLOR_BONUS = { maj7: 2.2, sus: 0.8, other: 1.5 };
  function colorBonusOf(tok) {
    if (tok[0] === "<" || tok === HOLD) return 0;
    if (/maj7|maj9|mM7/.test(tok)) return COLOR_BONUS.maj7;
    if (/sus/.test(tok)) return COLOR_BONUS.sus;
    if (/aug|add9|°|ø|dim|6|9|11|13|\//.test(tok)) return COLOR_BONUS.other;
    return 0;
  }
  function colorIndices(vocab) {
    const pairs = [];
    for (let i = 0; i < vocab.length; i++) {
      const b = colorBonusOf(vocab[i]);
      if (b > 0) pairs.push([i, b]);
    }
    return pairs;
  }
  // 曲内の彩り予算: 彩りコードは一度出ると文脈が再帰を呼び、ボーナスが自己強化して
  // 1曲まるごと maj7/sus ループになりやすい。コードチェンジ数に対する彩り比率が
  // RICH_TARGET を超えている間はボーナスを止め、曲単位の彩り率を安定させる。
  const RICH_TARGET = 0.15;
  function biasColor(y, pairs, richCount, evCount) {
    if (richCount > evCount * RICH_TARGET + 1) return y;   // 予算超過中はボーナス停止
    for (const [i, b] of pairs) y[i] += b;
    return y;
  }

  // ---- 音楽理論プライア（機能和声） ----
  // Transformer がデータから学んだ分布に、理論的に推奨される進行への軽いロジットボーナスを混ぜる。
  //  ・機能進行: T→S→D→T の循環を優遇し、D→S の逆行を抑制
  //  ・ドミナント解決: 属七(V7/II7/III7/VI7 等)は5度下の解決先を優遇（セカンダリードミナント対応）
  //  ・sus解決: sus は同根音の非susコードへの解決を優遇
  // 語彙ごとの分類は romanToChord で一度だけ前計算。重み THEORY_WEIGHT で効き具合を調整。
  const THEORY_WEIGHT = 0.5;   // 0=データのみ / 1=理論一辺倒に近い。0.5でデータよりやや理論寄り

  // 周期性バイアス: 4小節前(16拍)・8小節前(32拍)に鳴っていたトークンを優遇し、
  // 「さっきの進行に戻る」＝曲の反復構造を促す。データでは4小節ブロックの33%が反復に
  // 属するが、生成時の各種バイアスのゆらぎで自然な反復がほぼ出なくなるための補正。
  // 形式テンプレートではなくソフトな優遇なので、モデルが違う進行を選ぶ自由は残る。
  const PERIOD_BONUS = 1.25;
  function biasPeriodic(y, out, idxMap) {
    for (const L of [16, 32]) {
      if (out.length >= L) {
        const ti = idxMap[out[out.length - L]];
        if (ti != null) y[ti] += PERIOD_BONUS;
      }
    }
    return y;
  }
  const THEORY_TRANS = {                       // 機能間の推奨度（T=主, S=下属, D=属）
    "T,T": 0, "T,S": 0.4, "T,D": 0.3,
    "S,T": 0.2, "S,S": 0, "S,D": 0.5,
    "D,T": 0.7, "D,S": -0.7, "D,D": 0,
  };
  const FUNC_OF_ROOT = { 0: "T", 1: "S", 2: "S", 3: "T", 4: "T", 5: "S", 6: "D", 7: "D", 8: "S", 9: "T", 10: "S", 11: "D" };
  function theoryInfoOf(tok) {
    if (tok === HOLD || tok[0] === "<") return null;
    const { rootSemitone, typeKey } = romanToChord(tok);
    const isDom7 = /^dom/.test(typeKey);                       // 属七系（7/9/11/13）
    const isMajTriadSecondary = typeKey === "maj" && (rootSemitone === 2 || rootSemitone === 4 || rootSemitone === 9 || rootSemitone === 11);
    return {
      root: rootSemitone,
      func: FUNC_OF_ROOT[rootSemitone] || "T",
      resolveTo: isDom7 || isMajTriadSecondary ? (rootSemitone + 5) % 12 : null,
      resolveBonus: isDom7 ? 1.0 : 0.6,                        // 属七は強く、長三和音の二次ドミナントは控えめに
      isSus: typeKey === "sus4" || typeKey === "sus2",
    };
  }
  function theoryIndices(vocab) {
    const infos = [];
    for (let i = 0; i < vocab.length; i++) {
      const info = theoryInfoOf(vocab[i]);
      if (info) infos.push([i, info]);
    }
    return infos;
  }
  function biasTheory(y, infos, cur) {
    if (!cur) return y;                                        // 直前コード未確定（曲頭）は素通し
    for (const [i, c] of infos) {
      let b = THEORY_TRANS[cur.func + "," + c.func] || 0;
      if (cur.resolveTo != null && c.root === cur.resolveTo) b += cur.resolveBonus;
      if (cur.isSus && c.root === cur.root && !c.isSus) b += 0.8;
      if (b !== 0) y[i] += THEORY_WEIGHT * b;
    }
    return y;
  }

  // ---- ローマ数字 → {rootSemitone(主音からの半音), typeKey} ----
  const ROMAN_DEG = { I: 0, II: 1, III: 2, IV: 3, V: 4, VI: 5, VII: 6 };
  function romanToChord(roman) {
    let s = String(roman).trim();
    // 分数（スラッシュ）コード: '/半音(0-11)' を分離。bassSemitone は主音からの半音。
    let bassSemitone = null;
    const slash = s.indexOf("/");
    if (slash >= 0) { const bp = parseInt(s.slice(slash + 1), 10); if (!isNaN(bp)) bassSemitone = ((bp % 12) + 12) % 12; s = s.slice(0, slash); }
    let acc = 0;
    while (s[0] === "b" || s[0] === "#") { acc += s[0] === "#" ? 1 : -1; s = s.slice(1); }
    const m = s.match(/^(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)/);
    if (!m) return { rootSemitone: 0, typeKey: "maj", bassSemitone };
    const num = m[1];
    const isUpper = num === num.toUpperCase();
    const deg = ROMAN_DEG[num.toUpperCase()];
    const suffix = s.slice(num.length);
    const rootSemitone = (typeof MAJOR_SCALE !== "undefined" ? MAJOR_SCALE[deg] : [0, 2, 4, 5, 7, 9, 11][deg]) + acc;

    let typeKey;
    const low = suffix.toLowerCase();
    const hasDim = suffix.includes("°") || low.includes("dim");
    const hasHalf = suffix.includes("ø") || low.includes("m7b5") || low.includes("b5");
    if (low.includes("sus4") || low === "sus") typeKey = "sus4";
    else if (low.includes("sus2")) typeKey = "sus2";
    else if (hasHalf) typeKey = "m7b5";
    else if (hasDim) typeKey = "dim";
    else if (low.includes("aug") || suffix.includes("+")) typeKey = "aug";
    else if (low.includes("mm7") || suffix.includes("mM7")) typeKey = "mM7";
    else if (low.includes("add9")) typeKey = isUpper ? "add9" : "madd9";
    else if (low.includes("maj9") || low.includes("maj11") || low.includes("maj13")) typeKey = "maj9";
    else if (low.includes("maj7") || suffix.includes("M7")) typeKey = "maj7";
    else if (low.includes("13")) typeKey = isUpper ? "dom13" : "min11";
    else if (low.includes("11")) typeKey = isUpper ? "dom11" : "min11";
    else if (low.includes("m9")) typeKey = "min9";
    else if (low.includes("9")) typeKey = isUpper ? "dom9" : "min9";
    else if (low.includes("m7")) typeKey = "min7";
    else if (suffix.includes("7")) typeKey = isUpper ? "dom7" : "min7";
    else if (low.includes("6")) typeKey = isUpper ? "six" : "m6";
    else typeKey = isUpper ? "maj" : "min";
    return { rootSemitone: ((rootSemitone % 12) + 12) % 12, typeKey, bassSemitone };
  }

  class NeuralComposer {
    constructor(model) {
      this.model = model;
      const c = model.chord, mel = model.melody;
      this.special = model.special;
      // コードモデル。type=="transformer" なら自己注意モデル、そうでなければ LSTM。
      const cidx = Object.fromEntries(c.outVocab.map((t, i) => [t, i]));
      this.chordColorIdx = colorIndices(c.outVocab);   // 彩りコードの語彙インデックス（生成時ボーナス用）
      this.chordTheoryIdx = theoryIndices(c.outVocab); // 機能和声の語彙分類（理論プライア用）
      this.chordCatIdx = catCountsOf(c.outVocab);      // カテゴリ別索引（語彙偏り対策の予算用）
      if (c.type === "transformer") {
        this.chordTF = {
          dModel: c.dModel, nHeads: c.nHeads, nLayers: c.nLayers, maxPos: c.pos_emb.length,
          vocab: c.outVocab, idx: cidx, embTok: c.emb_tok, posEmb: c.pos_emb, layers: c.layers,
          lnfG: c.lnf_g, lnfB: c.lnf_b, Wout: c.Wout, bout: c.bout,
        };
        this.chord = { vocab: c.outVocab, idx: cidx };   // 互換用（idx.I など参照される）
      } else {
        this.chordTF = null;
        this.chord = {
          H: c.H, layers: toLayers(c), vocab: c.outVocab, emb: c.emb_tok,
          Wout: c.Wout, bout: c.bout, idx: cidx,
          embG: c.emb_g || null,
          gIdx: c.genreVocab ? Object.fromEntries(c.genreVocab.map((t, i) => [t, i])) : null,
        };
      }
      // メロディモデル。type=="transformer" なら自己注意モデル（コード文脈+拍位置つき）、そうでなければ LSTM。
      if (mel.type === "transformer") {
        this.melTF = {
          dModel: mel.dModel, nHeads: mel.nHeads, nLayers: mel.nLayers, maxPos: mel.pos_emb.length,
          vocab: mel.outVocab, idx: Object.fromEntries(mel.outVocab.map((t, i) => [t, i])),
          ctxVocab: mel.ctxVocab, ctxIdx: Object.fromEntries(mel.ctxVocab.map((t, i) => [t, i])),
          embTok: mel.emb_tok, embCtx: mel.emb_ctx, embBeat: mel.emb_beat, posEmb: mel.pos_emb,
          embPhr: mel.emb_phr || null,                 // 4小節フレーズ内の小節位置（旧モデルは無し）
          embG: mel.emb_g || null,
          gIdx: mel.genreVocab ? Object.fromEntries(mel.genreVocab.map((t, i) => [t, i])) : null,
          layers: mel.layers, lnfG: mel.lnf_g, lnfB: mel.lnf_b, Wout: mel.Wout, bout: mel.bout,
          restIdx: [], pitchIdx: {},                   // 休符トークン / 音程別トークン（生成時バイアス用）
        };
        mel.outVocab.forEach((t, i) => {
          if (t[0] === "R") this.melTF.restIdx.push(i);
          else {
            const off = parseInt(t, 10);
            if (!isNaN(off)) (this.melTF.pitchIdx[off] = this.melTF.pitchIdx[off] || []).push(i);
          }
        });
        this.mel = null;
      } else {
        this.melTF = null;
        this.mel = {
          H: mel.H, layers: toLayers(mel), vocab: mel.outVocab, embMel: mel.emb_mel, embChd: mel.emb_chd,
          Wout: mel.Wout, bout: mel.bout,
          ctxVocab: mel.ctxVocab,
          idx: Object.fromEntries(mel.outVocab.map((t, i) => [t, i])),
          ctxIdx: Object.fromEntries(mel.ctxVocab.map((t, i) => [t, i])),
          embG: mel.emb_g || null,
          gIdx: mel.genreVocab ? Object.fromEntries(mel.genreVocab.map((t, i) => [t, i])) : null,
        };
      }
      this.genres = c.genreVocab || null;   // 利用可能なジャンル一覧（UI用）
      // 強弱曲線モデル（任意）: 小節特徴 → 曲内zベロシティ。曲調に応じた盛り上げを学習済み
      const dy = model.dyn;
      this.dyn = dy ? { H: dy.H, feat: dy.feat, layers: toLayers(dy), Wout: dy.Wout, bout: dy.bout } : null;
      // 伴奏（コンプ）モデル（任意）：コード条件つきに16ステップの多声フレームを生成
      const cp = model.comp;
      this.comp = cp ? {
        H: cp.H, layers: toLayers(cp), nbins: cp.nbins, binLo: cp.binLo, steps: cp.steps,
        embChd: cp.emb_chd, Wout: cp.Wout, bout: cp.bout,
        ctxIdx: Object.fromEntries(cp.ctxVocab.map((t, i) => [t, i])),
      } : null;
    }

    static available() { return typeof window !== "undefined" && !!window.COMPOSER_MODEL; }
    hasComp() { return !!this.comp; }
    genreList() { return this.genres && this.genres.length > 1 ? this.genres.slice() : null; }

    // EOS（曲の終わり）までサンプルして「可変長」のコード進行を作る。曲の長さをモデルが決める。
    // 開始は I。minBars 未満では EOS を禁止し、maxBars で打ち切り、小節（4拍）単位に丸める。
    generateChordsUntilEnd(genre, temperature, minBars, maxBars) {
      minBars = minBars || 4; maxBars = maxBars || 16;
      if (this.chordTF) return this._genChordsTF(temperature, minBars, maxBars);
      const m = this.chord;
      const gv = genreVec(m, genre);
      const bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
      const minBeats = minBars * 4, maxBeats = maxBars * 4;
      // 開始コード: データの22%は短調(vi中心)なので、その比率で vi 始まりにする（短調曲の入口）
      const startIdx = (Math.random() < 0.22 && m.idx.vi != null) ? m.idx.vi : m.idx.I;
      let best = null;
      for (let attempt = 0; attempt < 4; attempt++) {
        const states = newStates(m.layers.length, m.H);
        let prev = bos; const out = [];
        let curChord = null, curInfo = null, run = 0;            // 現在のコードと持続拍数
        let evCount = 0, richCount = 0;                          // コードチェンジ数と彩りの数（予算制御）
        const tokCount = new Map(), catCount = { slash: 0, maj7: 0, sus: 0 };   // 語彙偏りの追跡
        let endedByEos = false;                                  // EOSで閉じたか（打ち切りとの区別）
        for (let i = 0; i < maxBeats; i++) {
          const h = runStack(concat(m.emb[prev], gv), states, m.layers);
          let idx;
          if (i === 0 && startIdx != null) idx = startIdx;       // トニック(I / 短調はvi)から始める
          else {
            const banned = out.length < minBeats ? new Set([bos, eos]) : new Set([bos]);
            let y = biasPeriodic(biasTheory(logits(h, m), this.chordTheoryIdx, curInfo), out, m.idx);
            y = biasDiversity(y, m.idx, this.chordCatIdx, tokCount, catCount, evCount);
            y = biasLongRun(biasWeakBeat(biasColor(y, this.chordColorIdx, richCount, evCount), m.idx[HOLD], i), m.idx[HOLD], m.idx[curChord], run);
            if (out.length >= minBeats) biasEos(y, eos, curChord);   // トニック上で終わりやすく
            idx = sample(y, temperature, banned);
            if (idx === eos) { endedByEos = true; break; }           // モデルが終わりと判断
          }
          const tok = m.vocab[idx];
          if (tok === HOLD || tok === curChord) run++;
          else {
            curChord = tok; curInfo = theoryInfoOf(tok); run = 1; evCount++;
            if (colorBonusOf(tok) > 0) richCount++;
            tokCount.set(tok, (tokCount.get(tok) || 0) + 1);
            if (tok.includes("/")) catCount.slash++;
            if (/maj7|maj9|mM7/.test(tok)) catCount.maj7++;
            if (/sus/.test(tok)) catCount.sus++;
          }
          out.push(tok); prev = idx;
        }
        let seq = expandHolds(out);                              // 保持トークンを拍コードに展開
        while (seq.length % 4 !== 0) seq.pop();                  // 小節単位に丸める
        if (!endedByEos) seq = trimToTonic(seq, minBeats);       // 打ち切り時は最後のトニック小節で閉じる
        if (seq.length >= minBeats && distinctChords(out) >= 4) { best = seq; break; }
        if (!best || seq.length > best.length) best = seq;
      }
      while (best.length < minBeats) best.push(best[best.length - 4] || "I");
      return best;
    }

    // コード進行（ローマ数字列）を生成。bars 個ちょうど（各小節の先頭拍コード）。開始I・終止Iを軽く強制。
    // 進行が単調（同じコードばかり）なら数回まで再サンプルして変化をつける。
    generateChords(bars, temperature, genre) {
      if (this.chordTF) {
        const m = this.chordTF, bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
        const cache = this._tfNewCache(), out = [];
        let y = this._tfStep(bos, cache);
        for (let i = 0; i < bars * 4; i++) {              // 拍単位でサンプル（保持トークン込み）
          let idx;
          if (i === 0 && m.idx.I != null) idx = m.idx.I;
          else idx = sample(biasWeakBeat(y, m.idx[HOLD], i), temperature, new Set([bos, eos]));
          out.push(m.vocab[idx]);
          y = this._tfStep(idx, cache);
        }
        const heads = expandHolds(out).filter((_, i) => i % 4 === 0);   // 小節頭のコードだけ返す
        if (m.idx.I != null && bars >= 1) heads[bars - 1] = "I";
        return heads;
      }
      let best = null, bestDistinct = -1;
      for (let attempt = 0; attempt < 5; attempt++) {
        const seq = this._sampleChords(bars, temperature, genre);
        const distinct = new Set(seq).size;
        if (distinct > bestDistinct) { bestDistinct = distinct; best = seq; }
        if (bars < 3 || distinct >= 3) break;                 // 3種以上あれば十分
      }
      return best;
    }

    // Transformer 前向き：ids（トークン列）→ 最後の位置の logits（次トークン分布）。
    _tfLogits(ids) {
      const m = this.chordTF, d = m.dModel, L = ids.length;
      let X = ids.map((id, p) => addv(m.embTok[id], m.posEmb[Math.min(p, m.maxPos - 1)]));
      for (const layer of m.layers) {
        const Xn = X.map((v) => layerNorm(v, layer.ln1_g, layer.ln1_b));
        const att = mhaCausal(Xn, layer, m.nHeads, d);
        for (let i = 0; i < L; i++) X[i] = addv(X[i], att[i]);
        const Xn2 = X.map((v) => layerNorm(v, layer.ln2_g, layer.ln2_b));
        for (let i = 0; i < L; i++) X[i] = addv(X[i], ffn(Xn2[i], layer));
      }
      const last = layerNorm(X[L - 1], m.lnfG, m.lnfB);
      const y = matVec(m.Wout, last); for (let k = 0; k < y.length; k++) y[k] += m.bout[k];
      return y;
    }

    // 自己回帰生成用の空KVキャッシュ（層ごとに K/V を蓄積）。位置カウンタも保持。
    _tfNewCache(m) {
      m = m || this.chordTF;
      return { pos: 0, K: m.layers.map(() => []), V: m.layers.map(() => []) };
    }
    // 入力ベクトル x を1ステップ投入して次分布のlogitsを返す（モデル汎用）。
    // K/Vをキャッシュに追記し O(履歴長) で処理（全再計算O(L²)を回避）。
    _tfStepX(m, x, cache) {
      const d = m.dModel;
      for (let l = 0; l < m.layers.length; l++) {
        const ly = m.layers[l];
        const xn = layerNorm(x, ly.ln1_g, ly.ln1_b);
        const q = addv(matVec(ly.Wq, xn), ly.bq);
        cache.K[l].push(addv(matVec(ly.Wk, xn), ly.bk));
        cache.V[l].push(addv(matVec(ly.Wv, xn), ly.bv));
        const att = attendOneQuery(q, cache.K[l], cache.V[l], m.nHeads, d);
        x = addv(x, addv(matVec(ly.Wo, att), ly.bo));
        const xn2 = layerNorm(x, ly.ln2_g, ly.ln2_b);
        x = addv(x, ffn(xn2, ly));
      }
      cache.pos++;
      const last = layerNorm(x, m.lnfG, m.lnfB);
      const y = matVec(m.Wout, last); for (let k = 0; k < y.length; k++) y[k] += m.bout[k];
      return y;
    }
    // コードモデル用: トークンIDから入力ベクトルを作って1ステップ。
    _tfStep(id, cache) {
      const m = this.chordTF;
      const x = addv(m.embTok[id], m.posEmb[Math.min(cache.pos, m.maxPos - 1)]);
      return this._tfStepX(m, x, cache);
    }

    // Transformer で EOS まで可変長のコード進行を生成（全履歴を自己注意で参照）。
    _genChordsTF(temperature, minBars, maxBars) {
      const m = this.chordTF;
      const bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
      const minBeats = minBars * 4, maxBeats = Math.min(maxBars * 4, m.maxPos - 2);
      // 開始コード: データの22%は短調(vi中心)なので、その比率で vi 始まりにする（短調曲の入口）
      const startIdx = (Math.random() < 0.22 && m.idx.vi != null) ? m.idx.vi : m.idx.I;
      let best = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        const cache = this._tfNewCache(), out = [];
        let y = this._tfStep(bos, cache);      // BOS投入 → 最初の分布
        let curChord = null, curInfo = null, run = 0;   // 現在のコードと持続拍数
        let evCount = 0, richCount = 0;        // コードチェンジ数と彩りの数（予算制御）
        const tokCount = new Map(), catCount = { slash: 0, maj7: 0, sus: 0 };   // 語彙偏りの追跡
        let endedByEos = false;                // EOSで閉じたか（打ち切りとの区別）
        for (let i = 0; i < maxBeats; i++) {
          let idx;
          if (i === 0 && startIdx != null) idx = startIdx;
          else {
            const banned = out.length < minBeats ? new Set([bos, eos]) : new Set([bos]);
            let yb = biasPeriodic(biasTheory(y, this.chordTheoryIdx, curInfo), out, m.idx);
            yb = biasDiversity(yb, m.idx, this.chordCatIdx, tokCount, catCount, evCount);
            yb = biasLongRun(biasWeakBeat(biasColor(yb, this.chordColorIdx, richCount, evCount), m.idx[HOLD], i), m.idx[HOLD], m.idx[curChord], run);
            if (out.length >= minBeats) biasEos(yb, eos, curChord);   // トニック上で終わりやすく
            idx = sample(yb, temperature, banned);
            if (idx === eos) { endedByEos = true; break; }
          }
          const tok = m.vocab[idx];
          if (tok === HOLD || tok === curChord) run++;
          else {
            curChord = tok; curInfo = theoryInfoOf(tok); run = 1; evCount++;
            if (colorBonusOf(tok) > 0) richCount++;
            tokCount.set(tok, (tokCount.get(tok) || 0) + 1);
            if (tok.includes("/")) catCount.slash++;
            if (/maj7|maj9|mM7/.test(tok)) catCount.maj7++;
            if (/sus/.test(tok)) catCount.sus++;
          }
          out.push(tok);
          y = this._tfStep(idx, cache);        // 次トークンをキャッシュに追記
        }
        let seq = expandHolds(out);            // 保持トークンを拍コードに展開
        while (seq.length % 4 !== 0) seq.pop();
        if (!endedByEos) seq = trimToTonic(seq, minBeats);   // 打ち切り時は最後のトニック小節で閉じる
        if (seq.length >= minBeats && distinctChords(out) >= 4) { best = seq; break; }
        if (!best || seq.length > best.length) best = seq;
      }
      while (best.length < minBeats) best.push(best[best.length - 4] || "I");
      return best;
    }

    _sampleChords(bars, temperature, genre) {
      const m = this.chord;
      const gv = genreVec(m, genre);
      const bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
      const banned = new Set([bos, eos]);   // 生成中は特殊トークンを禁止（長さを制御）
      const states = newStates(m.layers.length, m.H);
      let prev = bos;
      const out = [];
      for (let i = 0; i < bars * 4; i++) {                     // 拍単位でサンプル（保持トークン込み）
        const h = runStack(concat(m.emb[prev], gv), states, m.layers);
        let idx;
        if (i === 0 && m.idx.I != null) idx = m.idx.I;         // 主音から始める
        else idx = sample(biasWeakBeat(logits(h, m), m.idx[HOLD], i), temperature, banned);
        out.push(m.vocab[idx]);
        prev = idx;
      }
      const heads = expandHolds(out).filter((_, i) => i % 4 === 0);   // 小節頭のコードだけ返す
      if (m.idx.I != null && bars >= 1) heads[bars - 1] = "I";  // 終止感（トニックで終わる）
      return heads;
    }

    // 拍ごとのコード列 beatChords（長さ=4×小節数）に沿ってメロディを生成。
    // 各音の文脈は「その音が始まる拍のコード」＝小節の途中でコードが変わっても追従する。
    // 返り値: 各小節ごとの [{off,beats}|{rest,beats}]
    generateMelody(beatChords, temperature, beatsPerBar, genre) {
      if (this.melTF) return this._genMelodyTF(beatChords, temperature, beatsPerBar, genre);
      const m = this.mel;
      beatsPerBar = beatsPerBar || 4;
      const gv = genreVec(m, genre);
      const bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
      const banned = new Set([bos, eos]);
      const unk = m.ctxIdx[this.special.unk] != null ? m.ctxIdx[this.special.unk] : 0;
      const states = newStates(m.layers.length, m.H);
      let prev = bos;
      const bars = [];
      const nbars = Math.floor(beatChords.length / beatsPerBar);
      for (let bi = 0; bi < nbars; bi++) {
        const bc = beatChords.slice(bi * beatsPerBar, bi * beatsPerBar + beatsPerBar);
        let remaining = beatsPerBar, cur = 0;
        const notes = [];
        let guard = 0;
        while (remaining > 1e-6 && guard < 24) {
          guard++;
          const ch = bc[Math.min(beatsPerBar - 1, Math.floor(cur))];   // 現在の拍のコード
          // 分数コードが文脈語彙に無ければベースを外した素のコードで引く（それでも無ければUNK）
          let ctx = m.ctxIdx[ch];
          if (ctx == null && ch.includes("/")) ctx = m.ctxIdx[ch.split("/")[0]];
          if (ctx == null) ctx = unk;
          const x = concat(m.embMel[prev], m.embChd[ctx], gv);
          const h = runStack(x, states, m.layers);
          const idx = sample(logits(h, m), temperature, banned);
          prev = idx;
          const tok = m.vocab[idx];
          const parts = tok.split(":");
          let dur = parseFloat(parts[1]);
          if (!(dur > 0)) dur = 0.5;
          if (dur > remaining + 1e-6) dur = remaining;         // 小節をはみ出す音は詰める
          if (parts[0] === "R") notes.push({ rest: true, beats: dur });
          else notes.push({ off: parseInt(parts[0], 10), beats: dur });
          remaining -= dur; cur += dur;
        }
        if (remaining > 1e-6) notes.push({ rest: true, beats: remaining });
        bars.push(notes);
      }
      return bars;
    }

    // Transformer 版メロディ生成。入力 = 音トークン + コード文脈 + 小節内拍位置(16分単位)。
    // 自己注意で過去のフレーズを参照できるため、モチーフの再現・展開が出る。
    // 位置埋め込みの上限(maxPos)を超えたら直近 KEEP トークンでキャッシュを再構築（スライディングウィンドウ）。
    _genMelodyTF(beatChords, temperature, beatsPerBar, genre) {
      const m = this.melTF;
      beatsPerBar = beatsPerBar || 4;
      const bos = m.idx[this.special.bos], eos = m.idx[this.special.eos];
      const banned = new Set([bos, eos]);
      const unk = m.ctxIdx[this.special.unk] != null ? m.ctxIdx[this.special.unk] : 0;
      // ジャンル条件（無指定は pop）。混合コーパスのリズム様式ロックを防ぐ
      const gi = m.gIdx ? (m.gIdx[genre] != null ? m.gIdx[genre] : (m.gIdx.pop || 0)) : null;
      const gv = gi != null && m.embG ? m.embG[gi] : null;
      const KEEP = 96;
      let cache = this._tfNewCache(m);
      const recent = [];                                   // 再構築用の直近入力 [tok, ctx, beat, phr]
      const melInput = (tok, ctx, beat, phr, pos) => {
        let x = addv(addv(m.embTok[tok], m.embCtx[ctx]), addv(m.embBeat[beat], m.posEmb[Math.min(pos, m.maxPos - 1)]));
        if (m.embPhr) x = addv(x, m.embPhr[phr]);
        if (gv) x = addv(x, gv);
        return x;
      };
      const step = (tok, ctx, beat, phr) => {
        if (cache.pos >= m.maxPos - 1) {                   // ウィンドウを直近 KEEP 分で作り直す
          cache = this._tfNewCache(m);
          const replay = recent.splice(0, recent.length).slice(-KEEP);
          for (const [t, c, b, f] of replay) {
            this._tfStepX(m, melInput(t, c, b, f, cache.pos), cache);
            recent.push([t, c, b, f]);
          }
        }
        recent.push([tok, ctx, beat, phr]);
        return this._tfStepX(m, melInput(tok, ctx, beat, phr, cache.pos), cache);
      };
      let prev = bos;
      const bars = [];
      const nbars = Math.floor(beatChords.length / beatsPerBar);
      // 曲単位の偏り対策（Transformerは拾った様式に自己強化でロックしやすい）:
      //  ・休符予算: 休符がトークンの REST_TARGET を超えている間は休符にペナルティ（スカスカな曲を防ぐ）
      //  ・同音連打ペナルティ: 同じ音程が4回以上続いたら、その音程に漸増ペナルティ（一本調子を防ぐ）
      const REST_TARGET = 0.15;
      let restCount = 0, tokCount = 0, samePitchRun = 0, lastOff = null;
      for (let bi = 0; bi < nbars; bi++) {
        const bc = beatChords.slice(bi * beatsPerBar, bi * beatsPerBar + beatsPerBar);
        let remaining = beatsPerBar, cur = 0;
        const notes = [];
        let guard = 0;
        while (remaining > 1e-6 && guard < 24) {
          guard++;
          const ch = bc[Math.min(beatsPerBar - 1, Math.floor(cur))];   // 次に置く音の拍のコード
          let ctx = m.ctxIdx[ch];
          if (ctx == null && ch.includes("/")) ctx = m.ctxIdx[ch.split("/")[0]];
          if (ctx == null) ctx = unk;
          const beat = Math.round((cur % beatsPerBar) * 4) % 16;       // 小節内拍位置（16分単位）
          const y = step(prev, ctx, beat, bi % 4);                     // フレーズ内小節位置つき
          if (restCount > tokCount * REST_TARGET + 2) for (const ri of m.restIdx) y[ri] -= 2.0;
          if (samePitchRun >= 3 && lastOff != null)
            for (const pi of (m.pitchIdx[lastOff] || [])) y[pi] -= 1.5 * (samePitchRun - 2);
          const idx = sample(y, temperature, banned);
          prev = idx;
          const tok = m.vocab[idx];
          const parts = tok.split(":");
          let dur = parseFloat(parts[1]);
          if (!(dur > 0)) dur = 0.5;
          if (dur > remaining + 1e-6) dur = remaining;     // 小節をはみ出す音は詰める
          tokCount++;
          if (parts[0] === "R") {
            restCount++;
            notes.push({ rest: true, beats: dur });
          } else {
            const off = parseInt(parts[0], 10);
            if (off === lastOff) samePitchRun++; else { lastOff = off; samePitchRun = 1; }
            notes.push({ off, beats: dur });
          }
          remaining -= dur; cur += dur;
        }
        if (remaining > 1e-6) notes.push({ rest: true, beats: remaining });
        bars.push(notes);
      }
      return bars;
    }

    // 強弱曲線: 生成済みの進行とメロディから「小節ごとの強さ(曲内z)」を推論する。
    // POP909 の実演奏ベロシティから学習（曲調に応じて盛り上げ/落ち着きが変わる）。
    // 特徴ベクトルは tools/prepare_pop909_dyn.py の bar_features と完全に一致させること。
    dynamicsCurve(beatRomans, melBars) {
      const m = this.dyn;
      if (!m) return null;
      const RICH = /7|9|11|13|sus|aug|°|ø|6/;
      const states = newStates(m.layers.length, m.H);
      const curve = [];
      for (let bi = 0; bi < melBars.length; bi++) {
        const f = new Float64Array(m.feat);
        for (let k = 0; k < 4; k++) {
          const r = beatRomans[Math.min(beatRomans.length - 1, bi * 4 + k)];
          f[romanToChord(r).rootSemitone] += 0.25;
          const base = String(r).split("/")[0];
          const num = romanBaseOf(r).replace(/[b#]/g, "");
          if (num === num.toLowerCase()) f[12] += 0.25;
          if (RICH.test(base)) f[13] += 0.25;
        }
        const notes = (melBars[bi] || []).filter((n) => !n.rest);
        if (notes.length) {
          f[14] = Math.min(2, notes.length / 8);
          f[15] = notes.reduce((a, n) => a + n.off, 0) / notes.length / 24;
          f[16] = Math.max(...notes.map((n) => n.off)) / 24;
          f[18] = notes.some((n) => n.beats >= 2) ? 1 : 0;
        }
        const durSum = notes.reduce((a, n) => a + n.beats, 0);
        f[17] = Math.max(0, Math.min(1, (4 - durSum) / 4));
        f[19 + (bi % 4)] = 1;
        f[23] = Math.min(1, bi / 32);
        const h = runStack(f, states, m.layers);
        curve.push(matVec(m.Wout, h)[0] + m.bout[0]);
      }
      // 3点移動平均で滑らかに（小節単位のギザつきを抑える）
      return curve.map((v, i) => (curve[Math.max(0, i - 1)] + v + curve[Math.min(curve.length - 1, i + 1)]) / 3);
    }

    // 学習した伴奏モデルで、1コード分の伴奏イベント [{at:拍, midis:[…], dur:拍}] を生成。
    // roman=コード（文脈）, rootPc=そのコードの根音ピッチクラス（絶対音への変換用）。
    generateComp(roman, rootPc, temperature) {
      const m = this.comp;
      if (!m) return null;
      const unk = m.ctxIdx[this.special.unk] != null ? m.ctxIdx[this.special.unk] : 0;
      const ctx = m.ctxIdx[roman] != null ? m.ctxIdx[roman] : unk;
      const ctxVec = m.embChd[ctx];
      const heat = Math.max(0.4, Math.min(1.6, temperature || 0.9));
      const states = newStates(m.layers.length, m.H);
      let prev = new Float64Array(m.nbins);
      const rolls = [];
      for (let s = 0; s < m.steps; s++) {
        const x = new Float64Array(m.nbins + ctxVec.length);
        x.set(prev, 0); x.set(ctxVec, m.nbins);
        const h = runStack(x, states, m.layers);
        const y = logits(h, m);
        const frame = new Float64Array(m.nbins);
        const cand = [];
        for (let b = 0; b < m.nbins; b++) {
          const p = 1 / (1 + Math.exp(-y[b] / heat));
          if (Math.random() < p) cand.push([p, b]);
          else if (p > 0.4) cand.push([p, b]);          // 強い候補は温度に関わらず拾う
        }
        cand.sort((a, b) => b[0] - a[0]);
        for (const [, b] of cand.slice(0, 6)) frame[b] = 1;  // 1ステップ最大6声で濁りを防ぐ
        rolls.push(frame);
        prev = frame;
      }
      return this._rollToEvents(rolls, rootPc);
    }

    // 16ステップのピアノロール → ノートイベント（各ビンの連続ONを1音にまとめる＝持続/アルペジオ）
    _rollToEvents(rolls, rootPc) {
      const m = this.comp;
      const events = [];
      const stepBeats = 4 / m.steps;
      for (let b = 0; b < m.nbins; b++) {
        let s = 0;
        while (s < m.steps) {
          if (rolls[s][b]) {
            let e = s + 1; while (e < m.steps && rolls[e][b]) e++;
            const midi = 48 + rootPc + (m.binLo + b);
            if (midi >= 28 && midi <= 100) events.push({ at: s * stepBeats, midis: [midi], dur: (e - s) * stepBeats });
            s = e;
          } else s++;
        }
      }
      return events;
    }

    // コード列全体の伴奏を「状態を引き継いで」連続生成し、小節ごとのイベント列を返す。
    // 小節をまたいでLSTMの状態と直前フレームを保つので、グルーヴが途切れず一貫する。
    generateCompSequence(romans, rootPcs, temperature) {
      const m = this.comp;
      if (!m) return null;
      const unk = m.ctxIdx[this.special.unk] != null ? m.ctxIdx[this.special.unk] : 0;
      const heat = Math.max(0.4, Math.min(1.6, temperature || 0.9));
      const states = newStates(m.layers.length, m.H);
      let prev = new Float64Array(m.nbins);
      const perBar = [];
      for (let bi = 0; bi < romans.length; bi++) {
        const ctx = m.ctxIdx[romans[bi]] != null ? m.ctxIdx[romans[bi]] : unk;
        const ctxVec = m.embChd[ctx];
        const rolls = [];
        for (let s = 0; s < m.steps; s++) {
          const x = new Float64Array(m.nbins + ctxVec.length);
          x.set(prev, 0); x.set(ctxVec, m.nbins);
          const h = runStack(x, states, m.layers);
          const y = logits(h, m);
          const frame = new Float64Array(m.nbins);
          const cand = [];
          for (let b = 0; b < m.nbins; b++) {            // 純ベルヌーイ抽選＝音が付いたり消えたりして再発音（リズムの動き）が出る
            const p = 1 / (1 + Math.exp(-y[b] / heat));
            if (Math.random() < p) cand.push([p, b]);
          }
          cand.sort((a, b) => b[0] - a[0]);
          for (const [, b] of cand.slice(0, 6)) frame[b] = 1;
          rolls.push(frame);
          prev = frame;                                  // 直前フレームを次小節へ引き継ぐ
        }
        perBar.push(this._rollToEvents(rolls, rootPcs[bi]));
      }
      return perBar;
    }
  }

  window.NeuralComposer = NeuralComposer;
  window.romanToChord = romanToChord;
  window.romanBaseOf = romanBaseOf;      // 反復検出・セクション判定用（compose.js）
  window.isMinorRomans = isMinorRomans;  // 長調/短調（vi中心）判定用（compose.js）
})();
