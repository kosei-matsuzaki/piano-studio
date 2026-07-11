#!/usr/bin/env python3
# ============================================================================
# train.py — AI作曲用の小型LSTMを2つ学習し、ブラウザ用の重み(../model.js)を書き出す。
#
#   ① コード進行モデル : コード列(ローマ数字トークン)を学習し、次のコードを予測
#   ② メロディモデル   : 「直前の音」と「現在のコード」から次の音(音程+音価)を予測
#
# 依存は numpy のみ（PyTorchが入っていれば置き換え可能だが、ここは純numpyで完結）。
# BPTT + Adam を手書きで実装している。データが小さいのでCPUで数秒〜数十秒。
#
#   使い方:  python3 tools/train.py
#   出力:    model.js （index.html が <script> で読み込む）
# ============================================================================

import os, sys, json, math, re, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")
DATA_DIR = os.path.join(HERE, "data")          # 前処理スクリプトが出力する *.corpus 置き場
OUT = os.path.join(HERE, "..", "src", "model.js")

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
BOS, EOS, UNK = "<S>", "<E>", "<UNK>"

rng = np.random.default_rng(12345)


# ------------------------- コーパス読み込み ---------------------------------
def note_to_midi(name):
    m = re.match(r"^([A-G])([#b]?)(-?\d+)$", name)
    if not m:
        raise ValueError(f"bad note: {name}")
    shift = 1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0
    return NOTE_PC[m.group(1)] + (int(m.group(3)) + 1) * 12 + shift


def fmt_dur(d):
    d = round(float(d), 3)
    return str(int(d)) if d == int(d) else str(d)


HOLD = "_"          # 保持トークン: 「直前のコードを1拍のばす」


def to_holds(toks):
    """拍ごとのコード列を保持トークン形式に正規化（既に保持形式なら変化なし）。
       例: I I I I IV IV → I _ _ _ IV _"""
    out, prev = [], None
    for t in toks:
        if t == HOLD:
            out.append(HOLD)
            continue
        out.append(HOLD if t == prev else t)
        prev = t
    return out


def expand_holds(toks):
    """保持トークン形式 → 拍ごとのコード列（HOLD を直前コードに展開）。"""
    out, prev = [], "I"
    for t in toks:
        if t != HOLD:
            prev = t
        out.append(prev)
    return out


def parse_corpus(path):
    """1ファイルをパース。melody / genre / time は省略可（genre 無しは 'pop'、time 無しは '4/4' 扱い）。
       ブロック書式は `genre:`(任意) → `time:`(任意) → `key:` → `chords:` → `melody:`(任意)。
       chords は保持トークン形式（拍ごと・HOLD=保持）に正規化して返す。
       旧フォーマット（melody があり 1小節1コード）は 4拍に展開してから正規化する。"""
    songs = []
    cur = {}
    pending_genre = None
    pending_time = None

    def close(cur):
        if not cur.get("chords"):
            return
        chords, bars = cur["chords"], cur.get("melody")
        if bars and len(chords) == len(bars):           # 旧フォーマット(1小節1コード)は拍に展開
            chords = [c for c in chords for _ in range(4)]
        cur["chords"] = to_holds(chords)
        songs.append(cur)

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("genre:"):
                pending_genre = line[len("genre:"):].strip()
            elif line.startswith("time:"):
                pending_time = line[len("time:"):].strip()
            elif line.startswith("key:"):
                close(cur)
                cur = {"key": line[4:].strip(), "genre": pending_genre or "pop",
                       "time": pending_time or "4/4"}
                pending_genre = pending_time = None
            elif line.startswith("chords:"):
                cur["chords"] = line[len("chords:"):].split()
            elif line.startswith("melody:"):
                body = line[len("melody:"):].strip()
                cur["melody"] = [b.strip().split() for b in body.split("|")]
    close(cur)
    return songs


MIN_SLASH_COUNT = 30    # これ未満しか出ない分数（スラッシュ）トークンはベースを外して素のコードに


def _prune_rare_slash(songs):
    """レアな分数トークン（I6/2 等）を素のコード（I6）に落として語彙の肥大を防ぐ。
       よく出る転回・分数（I/4, V/2, IV/8 …）はそのまま学習される。"""
    from collections import Counter
    cnt = Counter(t for s in songs for t in s["chords"] if "/" in t)
    if not cnt:
        return
    pruned = {t for t, c in cnt.items() if c < MIN_SLASH_COUNT}
    n = 0
    for s in songs:
        ch = s["chords"]
        hit = False
        for i, t in enumerate(ch):
            if t in pruned:
                ch[i] = t.split("/")[0]; n += 1; hit = True
        if hit:
            s["chords"] = to_holds(ch)      # ベース省略で隣接同一コードができたら保持に再正規化
    kept = len(cnt) - len(pruned)
    print(f"分数コード: {kept} 種を保持 / {len(pruned)} 種({n}イベント)はベースを省略（<{MIN_SLASH_COUNT}回）")


def _dedup(songs):
    """完全重複エントリ（コード+メロディが同一）を除去。曲内の繰り返しセクションが
       同一チャンクとして複数回書き出されたものが主で、残すと学習分布を歪める。"""
    seen, out = set(), []
    for s in songs:
        key = (" ".join(s["chords"]),
               "|".join(" ".join(b) for b in s.get("melody") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    if len(out) < len(songs):
        print(f"重複除去: {len(songs) - len(out)} 曲を削除")
    return out


def load_all_corpora():
    """base の corpus.txt と tools/data/*.corpus をすべて読み込む。"""
    songs = parse_corpus(CORPUS)
    n_base = len(songs)
    extra_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.corpus")))
    for p in extra_files:
        s = parse_corpus(p)
        print(f"  + {os.path.relpath(p, HERE)}: {len(s)} 曲")
        songs += s
    print(f"読み込み: base={n_base}, 追加={len(songs) - n_base}, 合計={len(songs)}")
    songs = _dedup(songs)
    _prune_rare_slash(songs)
    return songs


def build_dataset(songs):
    """曲を学習用に変換。ジャンルも並列で返す。
       返り値: chord_seqs, mel_seqs, chord_genres, mel_genres
       （chord_seqs[i] のジャンルが chord_genres[i]。mel も同様。）"""
    chord_seqs, mel_seqs = [], []
    chord_genres, mel_genres = [], []
    for s in songs:
        tonic_pc = NOTE_PC[s["key"][0]]
        tonic_pc += 1 if s["key"][1:2] == "#" else -1 if s["key"][1:2] == "b" else 0
        tonic_ref = 60 + (tonic_pc % 12)
        genre = s.get("genre", "pop")
        chords = s["chords"]                            # 保持トークン形式（1トークン=1拍）
        bars = s.get("melody")
        if chords:
            chord_seqs.append(chords[:]); chord_genres.append(genre)
        if not bars:
            continue
        if len(chords) != 4 * len(bars):                # 拍コードは小節数×4
            print(f"  ! melody skip (chords {len(chords)} != 4×bars {4*len(bars)})", file=sys.stderr)
            continue
        beat_romans = expand_holds(chords)              # メロディ文脈は展開したコードで
        mel = []
        for bi, bar in enumerate(bars):
            beat_chords = beat_romans[bi * 4:bi * 4 + 4]   # この小節の4拍分のコード
            cur = 0.0
            for tok in bar:
                pitch, dur = tok.split(":")
                ctx = beat_chords[min(3, int(cur + 1e-6))]   # 音の開始拍のコードを文脈に
                if pitch == "R":
                    mtok = f"R:{fmt_dur(dur)}"
                else:
                    off = note_to_midi(pitch) - tonic_ref
                    mtok = f"{off}:{fmt_dur(dur)}"
                mel.append((mtok, ctx))
                cur += float(dur)
        mel_seqs.append(mel); mel_genres.append(genre)
    return chord_seqs, mel_seqs, chord_genres, mel_genres


# ------------------------------ LSTM ----------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class LSTMModel:
    """単層LSTM言語モデル。入力は1つ以上の埋め込みテーブルを連結して作る。
       emb_specs: [(name, vocab_size, dim), ...]  出力語彙 = out_vocab。"""

    def __init__(self, emb_specs, out_vocab, H):
        self.emb_specs = emb_specs
        self.H = H
        self.out_vocab = out_vocab
        Vout = len(out_vocab)
        D = sum(dim for _, _, dim in emb_specs)
        self.D = D
        p = {}
        for name, V, dim in emb_specs:
            p[f"emb_{name}"] = rng.normal(0, 0.1, (V, dim))
        # ゲート順序 [i, f, g, o]
        k = 1.0 / math.sqrt(H)
        p["Wih"] = rng.uniform(-k, k, (4 * H, D))
        p["Whh"] = rng.uniform(-k, k, (4 * H, H))
        p["b"] = np.zeros(4 * H)
        p["b"][H:2 * H] = 1.0  # forget gate バイアスを1に（学習安定）
        p["Wout"] = rng.uniform(-k, k, (Vout, H))
        p["bout"] = np.zeros(Vout)
        self.p = p

    def input_vec(self, ids):
        parts = []
        for (name, _, _), i in zip(self.emb_specs, ids):
            parts.append(self.p[f"emb_{name}"][i])
        return np.concatenate(parts)

    def loss_and_grad(self, seq):
        """seq: list of (input_ids_tuple, target_id). 1系列のBPTT。"""
        H = self.H
        p = self.p
        Wih, Whh, b, Wout, bout = p["Wih"], p["Whh"], p["b"], p["Wout"], p["bout"]
        T = len(seq)
        xs, hs, cs, gates = [], [], {-1: np.zeros(H)}, []
        c_prev = np.zeros(H)
        h_prev = np.zeros(H)
        hs_list = [h_prev]
        cs_list = [c_prev]
        loss = 0.0
        cache = []
        for t in range(T):
            ids, target = seq[t]
            x = self.input_vec(ids)
            z = Wih @ x + Whh @ h_prev + b
            i = sigmoid(z[0:H]); f = sigmoid(z[H:2 * H])
            g = np.tanh(z[2 * H:3 * H]); o = sigmoid(z[3 * H:4 * H])
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            logits = Wout @ h + bout
            logits -= logits.max()
            e = np.exp(logits); prob = e / e.sum()
            loss -= math.log(prob[target] + 1e-12)
            cache.append((ids, x, h_prev, c_prev, i, f, g, o, c, tc, h, prob, target))
            h_prev, c_prev = h, c

        # ---- backward ----
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(H)
        dc_next = np.zeros(H)
        for t in reversed(range(T)):
            ids, x, h_pr, c_pr, i, f, g, o, c, tc, h, prob, target = cache[t]
            dlogits = prob.copy(); dlogits[target] -= 1.0
            grads["Wout"] += np.outer(dlogits, h)
            grads["bout"] += dlogits
            dh = Wout.T @ dlogits + dh_next
            do = dh * tc; dzo = do * o * (1 - o)
            dc = dh * o * (1 - tc * tc) + dc_next
            df = dc * c_pr; dzf = df * f * (1 - f)
            di = dc * g; dzi = di * i * (1 - i)
            dg = dc * i; dzg = dg * (1 - g * g)
            dz = np.concatenate([dzi, dzf, dzg, dzo])
            grads["Wih"] += np.outer(dz, x)
            grads["Whh"] += np.outer(dz, h_pr)
            grads["b"] += dz
            dx = Wih.T @ dz
            dh_next = Whh.T @ dz
            dc_next = dc * f
            # 埋め込みへ勾配を分配
            off = 0
            for (name, _, dim), idx in zip(self.emb_specs, ids):
                grads[f"emb_{name}"][idx] += dx[off:off + dim]
                off += dim
        return loss, grads

    def export(self):
        p = self.p
        out = {"H": self.H, "outVocab": self.out_vocab,
               "Wih": p["Wih"].tolist(), "Whh": p["Whh"].tolist(),
               "b": p["b"].tolist(), "Wout": p["Wout"].tolist(),
               "bout": p["bout"].tolist(), "embDims": {}}
        for name, _, dim in self.emb_specs:
            out[f"emb_{name}"] = p[f"emb_{name}"].tolist()
            out["embDims"][name] = dim
        return out


def train(model, sequences, epochs, lr, clip=5.0, label=""):
    opt = Adam(model.p, lr=lr)
    for ep in range(epochs):
        order = rng.permutation(len(sequences))
        total = 0.0; ntok = 0
        for idx in order:
            seq = sequences[idx]
            loss, grads = model.loss_and_grad(seq)
            # 勾配クリップ
            gn = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if gn > clip:
                s = clip / (gn + 1e-8)
                for g in grads.values():
                    g *= s
            opt.step(grads)
            total += loss; ntok += len(seq)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            ppl = math.exp(total / max(1, ntok))
            print(f"  [{label}] epoch {ep+1}/{epochs}  loss/token={total/ntok:.3f}  ppl={ppl:.2f}")


# --------------------------- 語彙とデータ構築 --------------------------------
def make_vocab(tokens, specials):
    vocab = list(specials)
    for t in tokens:
        if t not in vocab:
            vocab.append(t)
    return vocab, {t: i for i, t in enumerate(vocab)}


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 進捗を随時フラッシュ
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--chord-epochs", type=int, default=0, help="0=データ量から自動")
    ap.add_argument("--mel-epochs", type=int, default=0, help="0=データ量から自動")
    ap.add_argument("--max-songs", type=int, default=0, help="デバッグ用: 使う曲数の上限(0=無制限)")
    args = ap.parse_args()

    songs = load_all_corpora()
    if args.max_songs:
        songs = songs[:args.max_songs]
        print(f"（--max-songs で {len(songs)} 曲に制限）")
    chord_seqs, mel_seqs, _cg, _mg = build_dataset(songs)   # numpy版はジャンル条件なし
    # データが多いほどエポックを減らす（1エポックが重くなるため）
    ce = args.chord_epochs or max(40, min(600, int(12000 / max(1, len(chord_seqs)))))
    me = args.mel_epochs or max(40, min(900, int(18000 / max(1, len(mel_seqs)))))

    # ---- コードモデル ----
    all_chords = [c for seq in chord_seqs for c in seq]
    cvocab, cidx = make_vocab(all_chords, [BOS, EOS])
    print(f"コード語彙: {len(cvocab)} 種  {cvocab}")
    chord_train = []
    for seq in chord_seqs:
        toks = [BOS] + seq + [EOS]
        ex = [((cidx[toks[t - 1]],), cidx[toks[t]]) for t in range(1, len(toks))]
        chord_train.append(ex)
    chord_model = LSTMModel([("tok", len(cvocab), 16)], cvocab, H=32)
    print(f"コードモデル学習: {len(chord_train)} 系列 × {ce} エポック")
    train(chord_model, chord_train, epochs=ce, lr=0.01, label="chord")

    # ---- メロディモデル ----
    all_mel = [m for seq in mel_seqs for (m, _) in seq]
    mvocab, midx = make_vocab(all_mel, [BOS, EOS])
    # メロディの文脈用コード語彙（UNK付き）
    ctx_chords = [c for seq in mel_seqs for (_, c) in seq]
    chvocab, chidx = make_vocab(ctx_chords, [UNK])
    print(f"メロディ語彙: {len(mvocab)} 種 / 文脈コード語彙: {len(chvocab)} 種")
    mel_train = []
    for seq in mel_seqs:
        mtoks = [BOS] + [m for (m, _) in seq] + [EOS]
        ctxs = [seq[0][1]] + [c for (_, c) in seq] + [seq[-1][1]]  # 各予測位置の文脈コード
        ex = []
        for t in range(1, len(mtoks)):
            ex.append(((midx[mtoks[t - 1]], chidx.get(ctxs[t], chidx[UNK])), midx[mtoks[t]]))
        mel_train.append(ex)
    mel_model = LSTMModel(
        [("mel", len(mvocab), 24), ("chd", len(chvocab), 16)], mvocab, H=64)
    print(f"メロディモデル学習: {len(mel_train)} 系列 × {me} エポック")
    train(mel_model, mel_train, epochs=me, lr=0.01, label="melody")

    # ---- 書き出し ----
    payload = {
        "version": 1,
        "meta": {"songs": len(songs), "chordVocab": len(cvocab), "melVocab": len(mvocab)},
        "special": {"bos": BOS, "eos": EOS, "unk": UNK},
        "chord": chord_model.export(),
        "melody": {**mel_model.export(), "ctxVocab": chvocab},
    }

    def round_floats(o, nd=5):
        if isinstance(o, float):
            return round(o, nd)
        if isinstance(o, list):
            return [round_floats(x, nd) for x in o]
        if isinstance(o, dict):
            return {k: round_floats(v, nd) for k, v in o.items()}
        return o

    payload = round_floats(payload)
    js = ("// model.js — tools/train.py が自動生成した学習済み重み。手で編集しないでください。\n"
          "// 学習データを増やすには tools/corpus.txt を編集して `python3 tools/train.py` を再実行。\n"
          "window.COMPOSER_MODEL = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(js)
    size = os.path.getsize(OUT) / 1024
    print(f"\n書き出し完了: {os.path.relpath(OUT, HERE)}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
