#!/usr/bin/env python3
# train_mel_tf.py — メロディモデルを decoder-only Transformer で学習し、model.js に差し替える。
#
# LSTM(直前の音+今のコード)ではフレーズを覚えられず通作的にさまよう
# （小節の完全一致反復率: データ2.78% vs 生成0.10%）。自己注意なら過去のフレーズ全体を
# 参照できるため、モチーフの再現・展開が学習できる（コードモデルで実証済みのアプローチ）。
#
#   入力(各位置 p): emb_tok(mel[p]) + emb_ctx(次に置く音の拍のコード)
#                   + emb_beat(次に置く音の小節内拍位置, 16分単位16バケット) + pos_emb(p)
#   出力: mel[p+1]（音トークン "音程:音価"）
#   ctx/beat を「ターゲット音の開始位置」で条件付けるのは train_torch.py の整合と同じ。
#
#   使い方:  python3 tools/train_mel_tf.py [--dim 128] [--layers 3] [--heads 4] [--epochs 40]
#   ブラウザ側 neural.js の Transformer 前向きと重みレイアウトを合わせてある。

import os, sys, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import train as T
from train_chord_tf import export_layer, r4

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JS = os.path.join(HERE, "..", "src", "model.js")
torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
MAXPOS = 192          # 8小節チャンク（最大〜128トークン）+ 余裕
NBEAT = 16            # 小節内の拍位置バケット（16分単位）


class MelTF(nn.Module):
    """ジャンル条件つき。混合コーパス(pop16分シンコペ/コラール四分/folkジグ)の
       リズム様式が無条件だと曲単位でモード崩壊するため、ジャンル埋め込みで分離する。"""
    def __init__(self, Vm, Vc, Vg, d, nhead, nlayers, dff):
        super().__init__()
        self.tok = nn.Embedding(Vm, d)
        self.ctx = nn.Embedding(Vc, d)
        self.beat = nn.Embedding(NBEAT, d)
        self.phr = nn.Embedding(4, d)        # 4小節フレーズ内の小節位置（呼吸・フレーズ終止の学習用）
        self.pos = nn.Embedding(MAXPOS, d)
        self.gen = nn.Embedding(Vg, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dff, dropout=0.1, activation="relu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, Vm)
        self.d = d

    def forward(self, x, ctx, beat, phrase, genre, key_pad):
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        h = (self.tok(x) + self.ctx(ctx) + self.beat(beat) + self.phr(phrase)
             + self.pos(pos)[None] + self.gen(genre)[:, None, :])
        cmask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.enc(h, mask=cmask, src_key_padding_mask=key_pad)
        return self.out(h)


def tok_dur(mtok):
    try:
        return float(mtok.split(":")[1])
    except (IndexError, ValueError):
        return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dff", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    songs = T.load_all_corpora()
    _, mel_seqs, _, mel_gen = T.build_dataset(songs)
    keep = [i for i, s in enumerate(mel_seqs) if len(s) + 2 <= MAXPOS]
    mel_seqs = [mel_seqs[i] for i in keep]; mel_gen = [mel_gen[i] for i in keep]
    from collections import Counter
    print(f"メロディ系列 {len(mel_seqs)} 本で学習（Transformer）ジャンル別: {dict(Counter(mel_gen))}", flush=True)
    mvocab, midx = T.make_vocab([m for s in mel_seqs for (m, _) in s], [T.BOS, T.EOS])
    chvocab, chidx = T.make_vocab([c for s in mel_seqs for (_, c) in s], [T.UNK])
    gvocab, gidx = T.make_vocab(mel_gen, ["pop"])
    unk = chidx[T.UNK]
    print(f"メロディ語彙: {len(mvocab)} / コード文脈語彙: {len(chvocab)} / ジャンル: {gvocab}  dim={args.dim} layers={args.layers}", flush=True)

    data = []
    for s, g in zip(mel_seqs, mel_gen):
        mtoks = [T.BOS] + [m for (m, _) in s] + [T.EOS]
        # ターゲット音（mtoks[1:]）の開始拍バケット・フレーズ内小節位置・文脈コード
        beats, phrs, ctxs, cur = [], [], [], 0.0
        for (m, c) in s:
            beats.append(int(round((cur % 4.0) * 4)) % NBEAT)
            phrs.append(int(cur // 4) % 4)
            ctxs.append(chidx.get(c, unk))
            cur += tok_dur(m)
        beats.append(int(round((cur % 4.0) * 4)) % NBEAT)     # EOS位置
        phrs.append(int(cur // 4) % 4)
        ctxs.append(ctxs[-1] if ctxs else unk)
        data.append((torch.tensor([midx[t] for t in mtoks[:-1]]),
                     torch.tensor(ctxs), torch.tensor(beats), torch.tensor(phrs),
                     torch.tensor([midx[t] for t in mtoks[1:]]), gidx[g]))

    lens = [len(x[0]) for x in data]
    order = sorted(range(len(data)), key=lambda i: lens[i])
    buckets = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]

    model = MelTF(len(mvocab), len(chvocab), len(gvocab), args.dim, args.heads, args.layers, args.dff)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for bk in np.random.permutation(len(buckets)):
            idx = buckets[bk]
            xb = pad_sequence([data[i][0] for i in idx], batch_first=True, padding_value=0)
            cb = pad_sequence([data[i][1] for i in idx], batch_first=True, padding_value=0)
            bb = pad_sequence([data[i][2] for i in idx], batch_first=True, padding_value=0)
            pb = pad_sequence([data[i][3] for i in idx], batch_first=True, padding_value=0)
            tb = pad_sequence([data[i][4] for i in idx], batch_first=True, padding_value=-100)
            gb = torch.tensor([data[i][5] for i in idx])
            key_pad = xb == 0
            key_pad[:, 0] = False                              # 先頭(BOS)は必ず有効
            logits = model(xb, cb, bb, pb, gb, key_pad)
            loss = lossf(logits.reshape(-1, logits.size(-1)), tb.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        if ep % max(1, args.epochs // 16) == 0 or ep == args.epochs - 1:
            print(f"  [melTF] epoch {ep+1}/{args.epochs}  loss={tot/nb:.3f}  ppl={math.exp(tot/nb):.2f}", flush=True)

    model.eval()

    def np_forward(mm, toks, ctxs, beats, phrs, g):           # neural.js と同じ演算（エクスポート検証用）
        d, nh = mm["dModel"], mm["nHeads"]; dh = d // nh
        E, C, B, P = (np.array(mm["emb_tok"]), np.array(mm["emb_ctx"]),
                      np.array(mm["emb_beat"]), np.array(mm["pos_emb"]))
        PH = np.array(mm["emb_phr"])
        G = np.array(mm["emb_g"])[g]
        X = np.stack([E[t] + C[c] + B[b] + PH[f] + G + P[p]
                      for p, (t, c, b, f) in enumerate(zip(toks, ctxs, beats, phrs))])
        L = X.shape[0]
        def ln(x, g, bb):
            mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(v + 1e-5) * np.array(g) + np.array(bb)
        for ly in mm["layers"]:
            Xn = ln(X, ly["ln1_g"], ly["ln1_b"])
            Q = Xn @ np.array(ly["Wq"]).T + ly["bq"]; K = Xn @ np.array(ly["Wk"]).T + ly["bk"]
            V = Xn @ np.array(ly["Wv"]).T + ly["bv"]
            out = np.zeros_like(X)
            for h in range(nh):
                q, k, v = Q[:, h*dh:(h+1)*dh], K[:, h*dh:(h+1)*dh], V[:, h*dh:(h+1)*dh]
                s = q @ k.T / math.sqrt(dh)
                s += np.triu(np.full((L, L), -1e30), 1)
                a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
                out[:, h*dh:(h+1)*dh] = a @ v
            X = X + out @ np.array(ly["Wo"]).T + ly["bo"]
            Xn2 = ln(X, ly["ln2_g"], ly["ln2_b"])
            X = X + np.maximum(Xn2 @ np.array(ly["W1"]).T + ly["b1"], 0) @ np.array(ly["W2"]).T + ly["b2"]
        X = ln(X, mm["lnf_g"], mm["lnf_b"])
        return X[-1] @ np.array(mm["Wout"]).T + mm["bout"]

    melody = {
        "type": "transformer", "dModel": args.dim, "nHeads": args.heads, "nLayers": args.layers,
        "outVocab": mvocab, "ctxVocab": chvocab, "genreVocab": gvocab,
        "emb_tok": r4(model.tok.weight.detach().numpy()),
        "emb_ctx": r4(model.ctx.weight.detach().numpy()),
        "emb_beat": r4(model.beat.weight.detach().numpy()),
        "emb_phr": r4(model.phr.weight.detach().numpy()),
        "emb_g": r4(model.gen.weight.detach().numpy()),
        "pos_emb": r4(model.pos.weight.detach().numpy()),
        "layers": [export_layer(l) for l in model.enc.layers],
        "lnf_g": r4(model.enc.norm.weight.detach().numpy()), "lnf_b": r4(model.enc.norm.bias.detach().numpy()),
        "Wout": r4(model.out.weight.detach().numpy()), "bout": r4(model.out.bias.detach().numpy()),
    }

    # エクスポート検証: PyTorch と numpy(=neural.js 相当) の logits を比較
    s0 = data[0]
    toks, ctxs, beats, phrs, g0 = (s0[0].tolist()[:24], s0[1].tolist()[:24],
                                   s0[2].tolist()[:24], s0[3].tolist()[:24], s0[5])
    with torch.no_grad():
        ref = model(torch.tensor([toks]), torch.tensor([ctxs]), torch.tensor([beats]),
                    torch.tensor([phrs]), torch.tensor([g0]),
                    torch.zeros(1, len(toks), dtype=torch.bool))[0, -1].numpy()
    got = np_forward(melody, toks, ctxs, beats, phrs, g0)
    err = float(np.max(np.abs(ref - got)))
    print(f"検証: PyTorch vs numpy(export) 最大誤差 = {err:.5f}  (小さければレイアウト正)", flush=True)

    txt = open(MODEL_JS, encoding="utf-8").read()
    key = "window.COMPOSER_MODEL = "
    i = txt.index(key) + len(key); j = txt.rindex("};") + 1
    obj = json.loads(txt[i:j])
    obj["melody"] = melody
    obj.setdefault("meta", {})["melVocab"] = len(mvocab)
    obj["meta"]["melArch"] = "transformer"
    out = ("// model.js — 学習済み重み（train_torch + train_comp + train_chord_tf + train_mel_tf）。手で編集しない。\n"
           "window.COMPOSER_MODEL = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n")
    open(MODEL_JS, "w", encoding="utf-8").write(out)
    print(f"\nmelody(Transformer) を差し替えました → model.js ({os.path.getsize(MODEL_JS)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
