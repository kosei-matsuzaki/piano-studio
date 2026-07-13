#!/usr/bin/env python3
# train_chord_tf.py — コード進行モデルを decoder-only Transformer で学習し、model.js に差し替える。
#
# LSTM は長距離を忘れるが、Transformer は自己注意で「それまでの全コード」を直接参照できるため、
# 反復・A/Bメロ・サビ・クライマックスといった曲構造を学習できる（LLMと同じ仕組み）。
#   ・入力: フル曲進行（メロディ無しのコード専用エントリ、拍ごとのローマ数字、キー正規化済み）
#   ・出力: model.js の "chord" ブロックを type="transformer" で置き換える（メロディはLSTMのまま）
#
#   使い方:  python3 tools/train_chord_tf.py [--dim 160] [--layers 3] [--heads 4] [--epochs 80]
#   ブラウザ側 neural.js の Transformer 前向きと重みレイアウトを合わせてある。

import os, sys, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JS = os.path.join(HERE, "..", "src", "model.js")
torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
MAXPOS = 280


class ChordTF(nn.Module):
    def __init__(self, V, d, nhead, nlayers, dff):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAXPOS, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dff, dropout=0.1, activation="relu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, V)
        self.d = d

    def forward(self, x, key_pad):
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        cmask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.enc(h, mask=cmask, src_key_padding_mask=key_pad)
        return self.out(h)


def r4(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


def export_layer(layer):
    """nn.TransformerEncoderLayer(norm_first=True) を neural.js 用に書き出す。"""
    d = layer.self_attn.embed_dim
    W = layer.self_attn.in_proj_weight.detach().numpy()      # [3d, d] = [Wq;Wk;Wv]
    b = layer.self_attn.in_proj_bias.detach().numpy()
    return {
        "Wq": r4(W[0:d]), "Wk": r4(W[d:2 * d]), "Wv": r4(W[2 * d:3 * d]),
        "bq": r4(b[0:d]), "bk": r4(b[d:2 * d]), "bv": r4(b[2 * d:3 * d]),
        "Wo": r4(layer.self_attn.out_proj.weight.detach().numpy()),
        "bo": r4(layer.self_attn.out_proj.bias.detach().numpy()),
        "ln1_g": r4(layer.norm1.weight.detach().numpy()), "ln1_b": r4(layer.norm1.bias.detach().numpy()),
        "ln2_g": r4(layer.norm2.weight.detach().numpy()), "ln2_b": r4(layer.norm2.bias.detach().numpy()),
        "W1": r4(layer.linear1.weight.detach().numpy()), "b1": r4(layer.linear1.bias.detach().numpy()),
        "W2": r4(layer.linear2.weight.detach().numpy()), "b2": r4(layer.linear2.bias.detach().numpy()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dff", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--genre-cap", type=int, default=2000, help="ジャンルごとの進行数上限(0=無制限)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    songs = T.load_all_corpora()
    pool = [(s["chords"], s.get("genre", "pop")) for s in songs
            if s.get("chords") and not s.get("melody") and len(s["chords"]) + 2 <= MAXPOS]
    # ジャンル別上限: songbook(Wikifonia 4千曲)が支配すると和声リズムが洋楽スタンダード様式
    # (1コード=1小節・拍間変化が減る)に寄るため、上限をかけて pop 系の比重を保つ。
    from collections import Counter, defaultdict
    if args.genre_cap:
        rng = np.random.RandomState(0)
        by_g = defaultdict(list)
        for i, (_, g) in enumerate(pool):
            by_g[g].append(i)
        keep = []
        for g, idxs in by_g.items():
            if len(idxs) > args.genre_cap:
                idxs = [idxs[k] for k in rng.permutation(len(idxs))[:args.genre_cap]]
            keep += idxs
        keep.sort()
        pool = [pool[i] for i in keep]
    seqs = [c for c, _ in pool]
    print(f"フル曲進行 {len(seqs)} 本で学習（Transformer）ジャンル別: {dict(Counter(g for _, g in pool))}", flush=True)
    cvocab, cidx = T.make_vocab([c for s in seqs for c in s], [T.BOS, T.EOS])
    print(f"コード語彙: {len(cvocab)}  dim={args.dim} layers={args.layers} heads={args.heads}", flush=True)

    data = [torch.tensor([cidx[t] for t in ([T.BOS] + s + [T.EOS])]) for s in seqs]
    lens = [len(x) for x in data]
    order = sorted(range(len(data)), key=lambda i: lens[i])
    buckets = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]

    model = ChordTF(len(cvocab), args.dim, args.heads, args.layers, args.dff)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for bk in np.random.permutation(len(buckets)):
            idx = buckets[bk]
            xb = pad_sequence([data[i] for i in idx], batch_first=True, padding_value=0)
            key_pad = xb == 0
            key_pad[:, 0] = False                              # 先頭(BOS)は必ず有効
            inp = xb[:, :-1]; tgt = xb[:, 1:].clone()
            tgt[tgt == 0] = -100                               # パディング位置は損失無視
            logits = model(inp, key_pad[:, :-1])
            loss = lossf(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        if ep % max(1, args.epochs // 16) == 0 or ep == args.epochs - 1:
            print(f"  [chordTF] epoch {ep+1}/{args.epochs}  loss={tot/nb:.3f}  ppl={math.exp(tot/nb):.2f}", flush=True)

    model.eval()

    def np_forward(ch, ids):                              # neural.js と同じ演算（エクスポート検証用）
        d, nh = ch["dModel"], ch["nHeads"]; dh = d // nh
        E, P = np.array(ch["emb_tok"]), np.array(ch["pos_emb"])
        X = np.stack([E[t] + P[p] for p, t in enumerate(ids)])
        ln = lambda v, g, b: (v - v.mean(-1, keepdims=True)) / np.sqrt(v.var(-1, keepdims=True) + 1e-5) * g + b
        for La in ch["layers"]:
            Xn = ln(X, np.array(La["ln1_g"]), np.array(La["ln1_b"]))
            Q = Xn @ np.array(La["Wq"]).T + np.array(La["bq"])
            K = Xn @ np.array(La["Wk"]).T + np.array(La["bk"])
            V = Xn @ np.array(La["Wv"]).T + np.array(La["bv"])
            Ln = len(ids); O = np.zeros((Ln, d))
            for i in range(Ln):
                for h in range(nh):
                    o = h * dh; q = Q[i, o:o + dh]
                    sc = np.array([q @ K[j, o:o + dh] for j in range(i + 1)]) / math.sqrt(dh)
                    sc = np.exp(sc - sc.max()); sc /= sc.sum()
                    O[i, o:o + dh] = sum(sc[j] * V[j, o:o + dh] for j in range(i + 1))
            X = X + (O @ np.array(La["Wo"]).T + np.array(La["bo"]))
            Xn2 = ln(X, np.array(La["ln2_g"]), np.array(La["ln2_b"]))
            ff = np.maximum(0, Xn2 @ np.array(La["W1"]).T + np.array(La["b1"])) @ np.array(La["W2"]).T + np.array(La["b2"])
            X = X + ff
        last = ln(X[-1], np.array(ch["lnf_g"]), np.array(ch["lnf_b"]))
        return last @ np.array(ch["Wout"]).T + np.array(ch["bout"])

    chord = {
        "type": "transformer", "dModel": args.dim, "nHeads": args.heads,
        "nLayers": args.layers, "dFF": args.dff, "outVocab": cvocab,
        "emb_tok": r4(model.tok.weight.detach().numpy()),
        "pos_emb": r4(model.pos.weight.detach().numpy()),
        "layers": [export_layer(l) for l in model.enc.layers],
        "lnf_g": r4(model.enc.norm.weight.detach().numpy()), "lnf_b": r4(model.enc.norm.bias.detach().numpy()),
        "Wout": r4(model.out.weight.detach().numpy()), "bout": r4(model.out.bias.detach().numpy()),
    }
    # 検証: エクスポート重みの numpy 前向き が PyTorch と一致するか（レイアウトの正しさ）
    ids = [cidx[t] for t in ([T.BOS] + seqs[0][:6])]
    with torch.no_grad():
        pt = model(torch.tensor([ids]), torch.zeros(1, len(ids), dtype=torch.bool))[0, -1].numpy()
    npf = np_forward(chord, ids)
    print(f"検証: PyTorch vs numpy(export) 最大誤差 = {np.abs(pt - npf).max():.5f}  (小さければレイアウト正)", flush=True)

    txt = open(MODEL_JS, encoding="utf-8").read()
    key = "window.COMPOSER_MODEL = "
    i = txt.index(key) + len(key); j = txt.rindex("};") + 1
    obj = json.loads(txt[i:j])
    obj["chord"] = chord
    obj.setdefault("meta", {})["chordVocab"] = len(cvocab)
    obj["meta"]["chordArch"] = "transformer"
    out = ("// model.js — 学習済み重み（train_torch=melody / train_chord_tf=chord Transformer）。手で編集しない。\n"
           "window.COMPOSER_MODEL = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n")
    open(MODEL_JS, "w", encoding="utf-8").write(out)
    print(f"\nchord(Transformer) を差し替えました → model.js ({os.path.getsize(MODEL_JS)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
