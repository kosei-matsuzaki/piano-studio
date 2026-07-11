#!/usr/bin/env python3
# train_dyn.py — 「小節ごとの強弱曲線」モデルを学習し、model.js に "dyn" ブロックとして注入する。
#
# 入力: prepare_pop909_dyn.py が作る data/pop909_dyn.npz（小節特徴24次元 → 曲内zベロシティ）
# モデル: 単層LSTM(H=48) + 線形回帰ヘッド。曲を小節列として流し、各小節の強さを回帰。
# 生成側: neural.js dynamicsCurve() が同じ特徴を生成物から計算して曲線を推論する。
#
#   使い方:  python3 tools/train_dyn.py [--epochs 30] [--hidden 48]

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from train_torch import export_lstm, r4

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "pop909_dyn.npz")
MODEL_JS = os.path.join(HERE, "..", "src", "model.js")
torch.manual_seed(0)
torch.set_num_threads(4)     # 小さいモデルは全コア並列だとスレッド競合で逆に遅い


class DynLM(nn.Module):
    def __init__(self, F, H):
        super().__init__()
        self.lstm = nn.LSTM(F, H, num_layers=1, batch_first=True)
        self.out = nn.Linear(H, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    d = np.load(NPZ)
    X, y, song = d["X"], d["y"], d["song"]
    F = X.shape[1]
    songs = []
    for sid in np.unique(song):
        m = song == sid
        songs.append((torch.tensor(X[m], dtype=torch.float32), torch.tensor(y[m], dtype=torch.float32)))
    rng = np.random.RandomState(0)
    order = rng.permutation(len(songs))
    n_val = max(1, len(songs) // 10)
    val = [songs[i] for i in order[:n_val]]
    trn = [songs[i] for i in order[n_val:]]
    print(f"学習 {len(trn)} 曲 / 検証 {n_val} 曲  特徴={F}次元 H={args.hidden}", flush=True)

    model = DynLM(F, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run(dataset, train):
        model.train(train)
        tot = n = 0
        idx = np.random.permutation(len(dataset)) if train else np.arange(len(dataset))
        for b0 in range(0, len(dataset), args.batch):
            bs = [dataset[i] for i in idx[b0:b0 + args.batch]]
            xb = pad_sequence([x for x, _ in bs], batch_first=True)
            yb = pad_sequence([t for _, t in bs], batch_first=True, padding_value=float("nan"))
            pred = model(xb)
            mask = torch.isfinite(yb)
            loss = ((pred[mask] - yb[mask]) ** 2).mean()
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * mask.sum().item(); n += mask.sum().item()
        return tot / n

    for ep in range(args.epochs):
        tr = run(trn, True)
        if ep % 5 == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                va = run(val, False)
            print(f"  [dyn] epoch {ep+1}/{args.epochs}  train_mse={tr:.3f}  val_mse={va:.3f}", flush=True)

    # 検証相関（曲線がどれだけ実演奏の起伏を説明するか）
    model.eval()
    cors = []
    with torch.no_grad():
        for x, t in val:
            p = model(x.unsqueeze(0))[0].numpy()
            m = np.isfinite(t.numpy())
            if m.sum() >= 8 and np.std(t.numpy()[m]) > 1e-6 and np.std(p[m]) > 1e-6:
                cors.append(np.corrcoef(p[m], t.numpy()[m])[0, 1])
    print(f"検証曲の強弱曲線との相関: 中央値 r={np.median(cors):.2f}（{len(cors)}曲）", flush=True)

    dyn = {"H": args.hidden, "feat": F, **export_lstm(model.lstm, model.out)}
    txt = open(MODEL_JS, encoding="utf-8").read()
    key = "window.COMPOSER_MODEL = "
    i = txt.index(key) + len(key); j = txt.rindex("};") + 1
    obj = json.loads(txt[i:j])
    obj["dyn"] = dyn
    obj.setdefault("meta", {})["dynCorr"] = round(float(np.median(cors)), 3)
    out = ("// model.js — 学習済み重み（chord/melody=Transformer, dyn=強弱曲線LSTM）。手で編集しない。\n"
           "window.COMPOSER_MODEL = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n")
    open(MODEL_JS, "w", encoding="utf-8").write(out)
    print(f"dyn を注入しました → model.js ({os.path.getsize(MODEL_JS)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
