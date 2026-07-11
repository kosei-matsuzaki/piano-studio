#!/usr/bin/env python3
# train_torch.py — PyTorch版トレーナー（大量データ・大きめモデル・多エポック向け）。
# 純numpy版 train.py と同じコーパス／同じ出力形式(model.js)を使うので、
# ブラウザ側(neural.js)は一切変更不要。学習だけ高速な torch に載せ替える。
#
#   使い方:  python3 tools/train_torch.py [--chord-epochs 200] [--mel-epochs 120]
#            [--chord-hidden 96] [--mel-hidden 160] [--batch 128] [--max-songs N]
#
# 単層LSTM（neural.js が単層前提のため）。隠れ次元と学習量を増やして表現力を上げる。

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import train as T          # コーパス読込・語彙作成・データ構築を再利用

torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = "cpu"


# ------------------------------- モデル -------------------------------------
GENRE_EMB = 8   # ジャンル埋め込み次元（Vg=0 のときジャンル条件なし）


class ChordLM(nn.Module):
    def __init__(self, V, E, H, nlayers=2, Vg=0):
        super().__init__()
        self.use_g = Vg > 0
        self.emb = nn.Embedding(V, E)
        if self.use_g:
            self.emb_g = nn.Embedding(Vg, GENRE_EMB)
        self.lstm = nn.LSTM(E + (GENRE_EMB if self.use_g else 0), H, num_layers=nlayers,
                            batch_first=True, dropout=0.2 if nlayers > 1 else 0.0)
        self.drop = nn.Dropout(0.15)
        self.out = nn.Linear(H, V)

    def forward(self, x, genre=None):
        e = self.emb(x)
        if self.use_g:
            ge = self.emb_g(genre).unsqueeze(1).expand(x.size(0), x.size(1), -1)
            e = torch.cat([e, ge], dim=-1)
        h, _ = self.lstm(e)
        return self.out(self.drop(h))


class MelodyLM(nn.Module):
    def __init__(self, Vm, Vc, Em, Ec, H, nlayers=2, Vg=0):
        super().__init__()
        self.use_g = Vg > 0
        self.emb_mel = nn.Embedding(Vm, Em)
        self.emb_chd = nn.Embedding(Vc, Ec)
        if self.use_g:
            self.emb_g = nn.Embedding(Vg, GENRE_EMB)
        self.lstm = nn.LSTM(Em + Ec + (GENRE_EMB if self.use_g else 0), H, num_layers=nlayers,
                            batch_first=True, dropout=0.2 if nlayers > 1 else 0.0)
        self.drop = nn.Dropout(0.15)
        self.out = nn.Linear(H, Vm)

    def forward(self, mel, ctx, genre=None):
        x = torch.cat([self.emb_mel(mel), self.emb_chd(ctx)], dim=-1)
        if self.use_g:
            ge = self.emb_g(genre).unsqueeze(1).expand(mel.size(0), mel.size(1), -1)
            x = torch.cat([x, ge], dim=-1)
        h, _ = self.lstm(x)
        return self.out(self.drop(h))


# ----------------------------- 学習ループ -----------------------------------
def make_buckets(lengths, bs):
    """系列を長さ順に並べて bs 個ずつのバケットに。パディングの無駄を大幅に減らす。"""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + bs] for i in range(0, len(order), bs)]


def epoch_batches(buckets):
    order = np.random.permutation(len(buckets))       # バケットの順序だけシャッフル
    for k in order:
        yield buckets[k]


def train_chord(model, ins, tgts, gen, epochs, bs, lr, clip=5.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    buckets = make_buckets([len(x) for x in ins], bs)
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for bidx in epoch_batches(buckets):
            xb = pad_sequence([ins[i] for i in bidx], batch_first=True, padding_value=0).to(DEV)
            yb = pad_sequence([tgts[i] for i in bidx], batch_first=True, padding_value=-100).to(DEV)
            gb = torch.tensor([gen[i] for i in bidx]).to(DEV)
            logits = model(xb, gb)
            loss = lossf(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
            tot += loss.item(); nb += 1
        if ep % max(1, epochs // 12) == 0 or ep == epochs - 1:
            print(f"  [chord] epoch {ep+1}/{epochs}  loss={tot/nb:.3f}  ppl={np.exp(tot/nb):.2f}", flush=True)


def train_melody(model, mel_in, ctx_in, tgts, gen, epochs, bs, lr, clip=5.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    buckets = make_buckets([len(x) for x in tgts], bs)
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for bidx in epoch_batches(buckets):
            mb = pad_sequence([mel_in[i] for i in bidx], batch_first=True, padding_value=0).to(DEV)
            cb = pad_sequence([ctx_in[i] for i in bidx], batch_first=True, padding_value=0).to(DEV)
            yb = pad_sequence([tgts[i] for i in bidx], batch_first=True, padding_value=-100).to(DEV)
            gb = torch.tensor([gen[i] for i in bidx]).to(DEV)
            logits = model(mb, cb, gb)
            loss = lossf(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
            tot += loss.item(); nb += 1
        if ep % max(1, epochs // 12) == 0 or ep == epochs - 1:
            print(f"  [melody] epoch {ep+1}/{epochs}  loss={tot/nb:.3f}  ppl={np.exp(tot/nb):.2f}", flush=True)


# ------------------------------- 書き出し -----------------------------------
def r4(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


def export_lstm(lstm, out_lin):
    """多層LSTMを layers[] 形式で書き出す（neural.js が層ごとに前向き）。"""
    layers = []
    for l in range(lstm.num_layers):
        Wih = getattr(lstm, f"weight_ih_l{l}").detach().numpy()
        Whh = getattr(lstm, f"weight_hh_l{l}").detach().numpy()
        b = (getattr(lstm, f"bias_ih_l{l}") + getattr(lstm, f"bias_hh_l{l}")).detach().numpy()
        layers.append({"Wih": r4(Wih), "Whh": r4(Whh), "b": r4(b)})
    return {"layers": layers,
            "Wout": r4(out_lin.weight.detach().numpy()), "bout": r4(out_lin.bias.detach().numpy())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chord-epochs", type=int, default=200)
    ap.add_argument("--mel-epochs", type=int, default=120)
    ap.add_argument("--chord-hidden", type=int, default=96)
    ap.add_argument("--mel-hidden", type=int, default=160)
    ap.add_argument("--chord-emb", type=int, default=32)
    ap.add_argument("--mel-emb", type=int, default=32)
    ap.add_argument("--ctx-emb", type=int, default=24)
    ap.add_argument("--chord-layers", type=int, default=2)
    ap.add_argument("--mel-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--max-songs", type=int, default=0)
    ap.add_argument("--mel-cap", type=int, default=0, help="メロディ系列の上限(0=全部)。学習時間の調整用。")
    ap.add_argument("--genre", action="store_true", help="ジャンル条件つきにする（既定=オフ＝全部まとめて学習）")
    args = ap.parse_args()

    songs = T.load_all_corpora()
    if args.max_songs:
        songs = songs[:args.max_songs]
    chord_seqs, mel_seqs, chord_gen, mel_gen = T.build_dataset(songs)
    if args.mel_cap:                                    # ジャンルごとに上限（多数派popを間引き均衡）
        from collections import defaultdict
        by_g = defaultdict(list)
        for i, g in enumerate(mel_gen):
            by_g[g].append(i)
        keep = []
        for g, idxs in by_g.items():
            if len(idxs) > args.mel_cap:
                idxs = [idxs[k] for k in np.random.permutation(len(idxs))[:args.mel_cap]]
            keep += idxs
        keep.sort()
        mel_seqs = [mel_seqs[i] for i in keep]; mel_gen = [mel_gen[i] for i in keep]
        print(f"メロディをジャンル別に最大 {args.mel_cap} へ均衡化", flush=True)
    print(f"コード系列 {len(chord_seqs)} / メロディ系列 {len(mel_seqs)}", flush=True)

    # ---- 語彙（ジャンル語彙も） ----
    cvocab, cidx = T.make_vocab([c for s in chord_seqs for c in s], [T.BOS, T.EOS])
    mvocab, midx = T.make_vocab([m for s in mel_seqs for (m, _) in s], [T.BOS, T.EOS])
    chvocab, chidx = T.make_vocab([c for s in mel_seqs for (_, c) in s], [T.UNK])
    gvocab, gidx = T.make_vocab(chord_gen + mel_gen, ["pop"])
    unk = chidx[T.UNK]
    Vg = len(gvocab) if args.genre else 0                # 0＝ジャンル条件なし
    from collections import Counter
    print(f"語彙: コード {len(cvocab)} / メロディ {len(mvocab)} / 文脈 {len(chvocab)}", flush=True)
    print(f"ジャンル条件: {'あり ' + str(gvocab) if args.genre else 'なし（全部まとめて学習）'}", flush=True)
    print(f"データ内訳（参考）コード {dict(Counter(chord_gen))} / メロディ {dict(Counter(mel_gen))}", flush=True)

    # ---- コード学習データ ----
    c_in, c_tgt, c_g = [], [], []
    for s, g in zip(chord_seqs, chord_gen):
        ids = [cidx[t] for t in ([T.BOS] + s + [T.EOS])]
        c_in.append(torch.tensor(ids[:-1])); c_tgt.append(torch.tensor(ids[1:])); c_g.append(gidx[g])
    chord_model = ChordLM(len(cvocab), args.chord_emb, args.chord_hidden, args.chord_layers, Vg).to(DEV)
    print(f"コードモデル: hidden={args.chord_hidden} × {args.chord_epochs}ep", flush=True)
    train_chord(chord_model, c_in, c_tgt, c_g, args.chord_epochs, args.batch, args.lr)

    # ---- メロディ学習データ ----
    m_in, ctx_in, m_tgt, m_g = [], [], [], []
    for s, g in zip(mel_seqs, mel_gen):
        mtoks = [T.BOS] + [m for (m, _) in s] + [T.EOS]
        ctxs = [s[0][1]] + [c for (_, c) in s] + [s[-1][1]]
        m_in.append(torch.tensor([midx[t] for t in mtoks[:-1]]))
        ctx_in.append(torch.tensor([chidx.get(ctxs[t], unk) for t in range(1, len(mtoks))]))
        m_tgt.append(torch.tensor([midx[t] for t in mtoks[1:]])); m_g.append(gidx[g])
    mel_model = MelodyLM(len(mvocab), len(chvocab), args.mel_emb, args.ctx_emb, args.mel_hidden, args.mel_layers, Vg).to(DEV)
    print(f"メロディモデル: hidden={args.mel_hidden} × {args.mel_epochs}ep", flush=True)
    train_melody(mel_model, m_in, ctx_in, m_tgt, m_g, args.mel_epochs, args.batch, args.lr)

    # ---- 書き出し（neural.js と同じレイアウト） ----
    chord_model.eval(); mel_model.eval()
    chord = {"H": args.chord_hidden, "outVocab": cvocab,
             "emb_tok": r4(chord_model.emb.weight.detach().numpy()),
             "embDims": {"tok": args.chord_emb}, **export_lstm(chord_model.lstm, chord_model.out)}
    melody = {"H": args.mel_hidden, "outVocab": mvocab, "ctxVocab": chvocab,
              "emb_mel": r4(mel_model.emb_mel.weight.detach().numpy()),
              "emb_chd": r4(mel_model.emb_chd.weight.detach().numpy()),
              "embDims": {"mel": args.mel_emb, "chd": args.ctx_emb},
              **export_lstm(mel_model.lstm, mel_model.out)}
    if args.genre:                                       # ジャンル条件つきのときだけ埋め込みを書き出す
        chord["genreVocab"] = gvocab; chord["emb_g"] = r4(chord_model.emb_g.weight.detach().numpy())
        chord["embDims"]["genre"] = GENRE_EMB
        melody["genreVocab"] = gvocab; melody["emb_g"] = r4(mel_model.emb_g.weight.detach().numpy())
        melody["embDims"]["genre"] = GENRE_EMB
    payload = {"version": 2,
               "meta": {"songs": len(songs), "chordSeqs": len(chord_seqs), "melSeqs": len(mel_seqs),
                        "chordVocab": len(cvocab), "melVocab": len(mvocab),
                        "genres": (gvocab if args.genre else None), "trainer": "torch"},
               "special": {"bos": T.BOS, "eos": T.EOS, "unk": T.UNK},
               "chord": chord, "melody": melody}

    js = ("// model.js — tools/train_torch.py が自動生成した学習済み重み。手で編集しないでください。\n"
          "// 再学習: python3 tools/train_torch.py （データは tools/corpus.txt と tools/data/*.corpus）\n"
          "window.COMPOSER_MODEL = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    with open(T.OUT, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"\n書き出し完了: {os.path.relpath(T.OUT, T.HERE)}  ({os.path.getsize(T.OUT)/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
