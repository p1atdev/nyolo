# nyolo

YOLO の PyTorch 再実装。

モデル定義、ラベル割り当て、損失関数、および推論パイプラインを提供します。

## 特徴

- **画面外キーポイント対応**: 画像境界で座標をクリップせず、画面外にはみ出した関節位置もそのまま学習・推論します。
- **NMS-free 推論**: 学習時は one-to-many と one-to-one の二重ヘッドを用い、推論時は後処理の NMS を不要にする one-to-one ヘッドのみで高速に出力します。
- **カスタマイズ可能な関節定義**: 標準の COCO 17 点（[`COCO17`](src/nyolo/pose/schema.py)）のほか、[`PoseSchema`](src/nyolo/pose/schema.py) で独自のキーポイント構造を定義できます。
- **対応環境**: Python 3.12 以上、CPU / Apple Silicon (MPS) の FP32 eager 実行。

## セットアップ

```sh
uv sync
```

## 使い方

### 1. 学習

[`PoseTrainingModule`](src/nyolo/learning/criterion.py) はモデルの順伝播、ラベル割り当て、損失計算をまとめたモジュールです。

```python
import torch
from nyolo import YOLO26Pose, YOLO26PoseConfig
from nyolo.learning import PoseTrainingModule
from nyolo.pose.types import PoseTargets

device = "mps"  # または "cpu"
training = PoseTrainingModule(YOLO26Pose(YOLO26PoseConfig(scale="s"))).to(device)
optimizer = torch.optim.AdamW(training.parameters(), lr=1e-3)

# images: float32 [B, 3, H, W] (RGB, 0.0〜1.0, H・Wは32の倍数)
# targets: PoseTargets (前処理後画像のピクセル座標系)
loss = training(images, targets, progress=epoch / (total_epochs - 1))

optimizer.zero_grad(set_to_none=True)
loss.total.backward()
optimizer.step()
```

教師データ [`PoseTargets`](src/nyolo/pose/types.py) には、バッチ内でパディングされた以下の Tensor（入力画像と同じ device）を渡します。

| フィールド | 形状 | 型 | 説明 |
|---|---|---|---|
| `boxes` | `[B, T, 4]` | `float32` | 画像内のバウンディングボックス (`xyxy`) |
| `labels` | `[B, T]` | `int64` | クラス ID |
| `keypoints` | `[B, T, K, 2]` | `float32` | キーポイント座標 (`xy`、負値や画像サイズ超過も可) |
| `visibility` | `[B, T, K]` | `int64` | `0`: 未定義/不明、`1`: 遮蔽または画面外、`2`: 可視 |
| `valid` | `[B, T]` | `bool` | パディングではなく実在するインスタンスか |

- `visibility > 0` のキーポイントが座標回帰損失の対象になります（画面外でも位置がわかる点は `visibility = 1` として座標を与えます）。
- `visibility = 0` の点は損失から除外されます（座標は `NaN` でも構いません）。

独自のキーポイント定義を使う場合は、`YOLO26PoseConfig(schema=...)` を `YOLO26Pose` に渡します。

### 2. 重みの保存と推論

推論用の重み保存には [`save_checkpoint`](src/nyolo/checkpoint.py)、推論の実行には [`Predictor`](src/nyolo/inference/predictor.py) を使用します。

```python
from PIL import Image
from nyolo.checkpoint import load_checkpoint, save_checkpoint
from nyolo.inference import Predictor

# 推論用チェックポイントの保存と読み込み
save_checkpoint("model.pt", training.model)
model = load_checkpoint("model.pt", device="mps")

# PIL Image からの推論（リサイズ・パディングと元画像座標への逆変換を自動実行）
predictor = Predictor(model, device="mps", image_size=(640, 640))
with Image.open("image.png") as image:
    result = predictor([image])

# 元画像のピクセル座標系での辞書リストを取得（画面外座標も保持）
records = result.to_records()
```

CLI から推論を実行することもできます。

```sh
uv run nyolo predict --checkpoint model.pt --input image.png \
  --device mps --size 640 --output predictions.json
```

## 開発・テスト

```sh
uv run pytest -q
uv run ruff check .
uv run ty check src
```

## 詳細ドキュメント・ライセンス

- [モデル・実装仕様](docs/specs/yolo26-pose.md)
- [動作検証レポート](docs/validation.md)
- [サイズ展開・torch.compile の検証結果と制約](docs/scale-compile-validation.md)
- [出典とライセンスの境界](docs/provenance.md)
- ライセンス: [Apache-2.0](LICENSE)
