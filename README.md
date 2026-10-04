# image-processing / sbsm

[English README](README.en.md)

SBSM (Statistical Background Subtraction Model) を CUDA で実行するサンプルです。
実装本体は sbsm フォルダ配下にあります。

## デモ結果

使用した入力画像:
- readme-images/sample.png

生成された出力画像:
- readme-images/sbsm_gpu.jpg

### 入力画像

![Input sample](readme-images/sample.png)

### 出力画像

![Output sbsm_gpu](readme-images/sbsm_gpu.jpg)

### 見どころ

- 赤色オーバーレイが前景候補の画素です。
- 背景として扱われる領域は元画像に近い見た目のまま残ります。
- ノイズが多い夜間シーンでも、変化領域を強調できることが確認できます。

## 数理モデル (統計学的背景差分法)

この実装は、各画素の観測RGB値 $I=(I_R,I_G,I_B)$ に対して、背景/前景の2クラスをベイズ判定します。

- 背景クラス: $w_0$
- 前景クラス: $w_1$
- 事前確率: $P(w_0)=0.95$, $P(w_1)=0.05$

背景尤度は、画素位置 $(x,y)$ と色チャンネル $c \in \{R,G,B\}$ ごとのヒストグラムから求めます。RGB各チャンネルは独立と仮定します。

$$
P(I\mid w_0) = \prod_{c\in\{R,G,B\}}
\frac{\operatorname{hist}[x,y,c,I_c]}{\operatorname{hist\_sum}[x,y,c]}
$$

前景尤度は、各チャンネルの8ビット値（0〜255）が一様に現れると近似します。

$$
P(I\mid w_1) = \left(\frac{1}{256}\right)^3
$$

観測 $I$ の確率は以下です。

$$
P(I) = P(w_0)P(I\mid w_0) + P(w_1)P(I\mid w_1)
$$

背景と前景は互いに排他的で、すべての画素をどちらかに分類するため、観測 $I$ に対する事後確率の和は1です。ベイズの定理を代入すると、

$$
\begin{aligned}
1 &= P(w_0\mid I) + P(w_1\mid I) \\
	&= \frac{P(I\mid w_0)P(w_0)}{P(I)}
	 + \frac{P(I\mid w_1)P(w_1)}{P(I)}
\end{aligned}
$$

両辺に $P(I)$ を掛けると、上の式が得られます。

事後確率はベイズの定理で求めます。

$$
P(w_k\mid I) = \frac{P(I\mid w_k)P(w_k)}{P(I)},
\qquad k\in\{0,1\}
$$

判定条件は次の通りです。

- 前景 if $P(w_1\mid I) > P(w_0\mid I)$

この条件を満たす画素を前景として扱い、出力画像では赤色でマーキングします。

## 主要ファイル

- メインスクリプト: sbsm/src/cuda_sbsm.py
- 入力画像: sbsm/data/input/
- 出力画像: sbsm/data/output/
- 背景ヒストグラム: sbsm/data/hist_xy.npy

## 実行方法 (Podman Compose)

1. sbsm ディレクトリへ移動

```bash
cd sbsm
```

2. 開発用コンテナをビルド

```bash
podman-compose -f docker-compose.dev.yml build
```

3. 対話実行

```bash
podman-compose -f docker-compose.dev.yml run --rm sbsm bash
python3 src/cuda_sbsm.py
```

## GPU 確認

sbsm 配下で次を実行:

```bash
podman-compose -f docker-compose.dev.yml run --rm sbsm bash -lc "nvidia-smi -L && python3 -c 'from numba import cuda; print(cuda.is_available())'"
```

期待値:
- nvidia-smi -L で GPU 名が表示される
- cuda.is_available() が True になる

## 補足

- cuda_sbsm.py は既定で sbsm/data/hist_xy.npy を読み込みます。
- 背景フレームからヒストグラムを再生成するコードはファイル内にありますが、現状コメントアウトされています。

