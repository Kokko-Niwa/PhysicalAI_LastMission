# AI仕分けブルドーザー（PAI最終課題）

Physical AI 応用1講座 最終課題の提出コードです。
Gazebo 上のブルドーザー型ロボットが、出てくるブロックを前カメラで見て、**CLIP（学習なし・推論のみ）** で種類を判定し、対応するゾーンへ押して仕分けます。

## 何をしているか

1. 判定台に新しいブロックが現れる（色の正解はロボットに教えない）
2. ロボットの前カメラ画像を CLIP に入力し、言葉で書いた仕分けルールと照らし合わせる
3. 一番近いルールに対応するゾーンを選ぶ
4. ブロックの斜め後ろに回り込み、ゾーンへ向かってまっすぐ押す
5. バックで戻り、次のブロックを待つ

### AIの使い方：言葉で書いたルールで仕分ける
`sorter.py` の先頭にある `RULES` に「こういう見た目なら、このゾーンへ」を英文で書きます。

```python
RULES = [
    ("a photo of a red box", "red_zone"),
    ("a photo of a blue box", "blue_zone"),
]
```

CLIP は画像と各文の類似度を出すだけなので、追加学習なしで、文を書き換えれば仕分けの基準を変えられます（zero-shot 分類）。

### 工夫点：Uブレード化
最初は平らな排土板だけでした。ですがこの状態で斜め方向から押すと、ブロックに角から当たるためか、押している途中でブロックが横へ滑り、ゾーンまで届きませんでした。
そこで、実機のブルドーザーでも運搬用に使われる **Uブレード**（両端が前に突き出た排土板）にしました。両脇にガードを付けたことで、ブロックがゾーンまで運ばれるようになりました。

## 環境

- Windows 11 + WSL2 / Ubuntu 24.04
- ROS 2 Jazzy + Gazebo Harmonic（`ros-jazzy-ros-gz`）
- Python 3.12、PyTorch、open_clip（ViT-B-32 / OpenAI 事前学習重み）
- GPU: NVIDIA RTX 4080（CPU でも動作可）

## セットアップ

```bash
sudo apt install -y ros-jazzy-ros-gz python3-numpy python3-opencv python3-venv ffmpeg
python3 -m venv --system-site-packages ~/pai_venv
source ~/pai_venv/bin/activate
pip install "numpy<2" torch torchvision open_clip_torch
```

## 実行（ターミナル4つ）

```bash
# 1. シミュレータ
gz sim -r sorting_arena.sdf
# 2. Gazebo と ROS 2 の橋
bash run_bridge.sh
# 3. 録画（俯瞰カメラ＋ロボ目線ワイプ＋判定字幕）
python3 recorder.py --no-preview
# 4. 本体（pai_venv の中で）
python3 sorter.py
```

判定時の画像と結果は `log/`、動画は `video/` に保存されます。

## ファイル

| ファイル | 役割 |
|---|---|
| `sorting_arena.sdf` | ワールド：Uブレード付き差動二輪ブルドーザー、前カメラ、仕分けゾーン、俯瞰カメラ |
| `sorter.py` | 本体：ブロック生成 → CLIP判定 → 押す、の繰り返しと結果の記録 |
| `motion.py` | 移動制御：その場旋回・目標点への走行・バック |
| `recorder.py` | 俯瞰カメラ映像の録画（ワイプと字幕つき） |
| `run_bridge.sh` | ros_gz_bridge の起動 |
| `snapshot.py` | カメラ画像を1枚保存する確認用 |
| `clip_test.py` | 保存した画像を CLIP で判定する確認用 |

## 動画

YouTube: [https://youtu.be/aDikn_-uflo](https://youtu.be/aDikn_-uflo)
※ 2倍速で編集しています。
