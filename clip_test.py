#!/usr/bin/env python3
"""
clip_test.py  —  CLIP（画像と言葉を照らし合わせるAI）でブロックを判定してみる確認用スクリプト

使い方（pai_venv に入った状態で）:
    python3 ~/pai_final/clip_test.py ~/pai_final/snap_front.png

仕組み:
    下の RULES に「こういう見た目なら、このゾーンへ」を言葉で書いておく。
    CLIP は画像と各説明文の「似ている度合い」を出すので、一番似ている説明文のゾーンを選ぶ。
    学習は一切していない（推論だけ）。ルールを書き換えれば仕分けの基準が変わる。
"""
import sys

import open_clip
import torch
from PIL import Image

# 仕分けルール：(CLIPに見せる説明文[英語], 送り先ゾーン)
RULES = [
    ("a photo of a red box", "red_zone"),
    ("a photo of a blue box", "blue_zone"),
]

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


@torch.no_grad()
def classify(model, preprocess, tokenizer, device, pil_image, rules):
    image = preprocess(pil_image).unsqueeze(0).to(device)
    texts = tokenizer([r[0] for r in rules]).to(device)
    img_f = model.encode_image(image)
    txt_f = model.encode_text(texts)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_f @ txt_f.T).softmax(dim=-1)[0].tolist()
    best = max(range(len(rules)), key=lambda i: probs[i])
    return best, probs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "snap_front.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使うデバイス: {device}")
    print("CLIPを読み込み中…（初回はモデルのダウンロードで少し待ちます）")
    model, preprocess, tokenizer = load_clip(device)

    best, probs = classify(model, preprocess, tokenizer, device, Image.open(path).convert("RGB"), RULES)
    print(f"\n画像: {path}")
    for (text, zone), p in zip(RULES, probs):
        mark = "  ←これ！" if (text, zone) == RULES[best] else ""
        print(f"  {p*100:5.1f}%  {text}  → {zone}{mark}")
    print(f"\n判定: {RULES[best][1]} へ運びます")


if __name__ == "__main__":
    main()
