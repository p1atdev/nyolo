# nyolo

独立実装の PyTorch YOLO26-Pose ライブラリ。初期版は **n サイズ・pose・画面外
キーポイント・CPU/MPS の FP32 eager** に対応します。ソースコードは Apache-2.0。

モデル、割り当て、損失、推論を提供します。Dataset と実運用の学習ループは含めず、
必要になった時点で別パッケージ `nyolo-train` に追加する構成です。
事前学習済み重みは同梱・自動取得しません。実用的な推論には自身で学習した重みが必要です。

## セットアップと検証

```sh
uv sync --frozen
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen ty check src
```

MPS の実機で、合成した2画像への過学習、画面外点の誤差、forward/backward、
640×640 の推論、重みの再読込、PIL 推論 API をまとめて検証できます。
この検証用ループは配布パッケージの外にあります。

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/validate_pose.py \
  --device mps --steps 240 --output artifacts/mps-smoke
```

`metrics.json` に合否と測定値、`predictions.json` に画面外座標を含む予測を保存します。
生成する重みは合成データへの過学習確認専用です。実画像への精度を示すものではありません。
MPS 未対応演算を CPU に回す設定や、device の自動切替は使いません。

## 学習コードから使う

```python
import torch
from nyolo import YOLO26Pose
from nyolo.learning import PoseTrainingModule
from nyolo.pose.types import PoseTargets

training = PoseTrainingModule(YOLO26Pose()).to("mps")
optimizer = torch.optim.AdamW(training.parameters(), lr=0.001)

# images: float32 [B,3,H,W], RGB/255, H/W は32の倍数
# targets: PoseTargets。座標は前処理後の画像のピクセル単位。
loss = training(images, targets, progress=epoch / (total_epochs - 1))
optimizer.zero_grad(set_to_none=True)
loss.total.backward()
optimizer.step()
```

`PoseTargets` は次の padded tensor を受け取ります。全 tensor は入力と同じ device に置きます。

| フィールド | 形状 | 型・意味 |
|---|---|---|
| `boxes` | B,T,4 | float32、画像内の xyxy bbox |
| `labels` | B,T | int64、クラス ID |
| `keypoints` | B,T,K,2 | float32、画面外を許容する xy |
| `visibility` | B,T,K | int64、0=位置不明、1=遮蔽、2=可視 |
| `valid` | B,T | bool、実在するインスタンス |

画面外点は座標を保って `visibility=1` とします。`v>0` の全点を座標損失の対象にし、
画像境界でクリップしません。`v=0` の座標は NaN でも受け付け、損失から除外します。
キーポイントスコアは位置推定の可否で学習し、物理的な可視性とは区別します。

標準は COCO 17 点。独自の関節数は `PoseSchema` に関節名、左右対応、OKS 定数を明示し、
`YOLO26PoseConfig(schema=...)` で指定できます。

## 推論と保存

```python
from PIL import Image
from nyolo.checkpoint import save_checkpoint, load_checkpoint, save_training_state
from nyolo.inference import Predictor

save_checkpoint("model.pt", training.model)
save_training_state("training-components.pt", training)  # RLE の学習済み状態も保存

model = load_checkpoint("model.pt", device="mps")
predictor = Predictor(model, device="mps", image_size=(640, 640))
with Image.open("image.png") as image:
    result = predictor([image])
records = result.to_records()  # 元画像の座標。負の値や画像寸法超過を保持。
```

`Predictor` は渡されたモデルを指定 device の eval モードにします。学習を続ける際は
`training.train()` を呼んでください。推論は one-to-one / NMS-free の経路です。
モデルを直接呼ぶ `model(images)` は生 tensor、`model.forward_train(images)` は二系統の
生 tensor を返します。RLE flow は推論時に実行しません。

```sh
uv run --frozen nyolo predict --checkpoint model.pt --input image.png \
  --device mps --size 640 --output predictions.json
```

保存形式は version・構造・関節 schema・座標規約・state_dict を持ち、読込は
`weights_only=True` / strict です。`training-components.pt` はモデルと criterion の
保存であり、optimizer や RNG を含む完全な学習再開ファイルではありません。

## 実装範囲と来歴

公開論文から構成した独立実装です。Ultralytics の実装、モデル YAML、公式重みを
取り込まず、公式 checkpoint の互換性・同一パラメータ数・公式精度の再現は保証しません。
公開論文に詳細がない tower 幅・正規化等の選択は、独自の構造バージョンとして記録します。

- [実装仕様](docs/specs/yolo26-pose.md)
- [MPS の検証結果](docs/validation.md)
- [出典・ライセンスの境界](docs/provenance.md)
- [Apache-2.0 ライセンス](LICENSE)

CUDA、torch.compile、Triton、他サイズ・他タスク、TPU は未実装です。
コードのライセンスと、利用するデータ・配布する重み・第三者の権利は別途確認してください。
