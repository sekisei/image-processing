# SBSM (CUDA)

SBSM を CUDA + Docker Compose で実行するための最小構成です。

## 概要

- 実行スクリプト: src/cuda_sbsm.py
- 入力画像の既定値: data/input/IMAG1138.jpg
- 出力画像の既定値: data/output/sbsm_gpu.jpg
- ヒストグラム入力の既定値: data/hist_xy.npy

実行時のパスは環境変数で上書きできます。

- SBSM_HIST_PATH
- SBSM_INPUT_IMAGE
- SBSM_OUTPUT_IMAGE

## ディレクトリ構成

- .dockerignore
- Dockerfile.dev
- Dockerfile.prod
- docker-compose.dev.yml
- docker-compose.prod.yml
- requirements.txt
- src/
  - cuda_sbsm.py
- data/
  - input/
  - output/

## 前提条件

- Docker Engine
- Docker Compose v2
- NVIDIA Driver
- NVIDIA Container Toolkit

GPU を使うため、ホスト側で NVIDIA ランタイムが有効である必要があります。

## 開発用 (dev)

特徴:

- ソースを共有マウント (.:/workspace)
- ローカル編集をコンテナへ即時反映

実行:

1. ディレクトリ移動
   cd /home/manager/workspace/image-processing/SBSM
2. ビルドして起動
   docker compose -f docker-compose.dev.yml up --build

## 本番想定 (prod)

特徴:

- 共有マウントなし
- イメージ内のソースで実行
- restart: unless-stopped

実行:

1. ディレクトリ移動
   cd /home/manager/workspace/image-processing/SBSM
2. ビルドしてデタッチ起動
   docker compose -f docker-compose.prod.yml up --build -d

停止:

- dev: docker compose -f docker-compose.dev.yml down
- prod: docker compose -f docker-compose.prod.yml down

## 入出力ファイル

- 入力画像: data/input/IMAG1138.jpg
- 出力画像: data/output/sbsm_gpu.jpg
- ヒストグラム: data/hist_xy.npy

注意:

- data/hist_xy.npy が存在しない場合、実行時に読み込みエラーになります。
- 別ファイルを使う場合は Compose の environment を変更してください。

## 数理モデル (統計学的背景差分法)

この実装は、各画素の観測輝度 I に対して背景/前景の 2 クラスをベイズ判定します。

- 背景クラス: w0
- 前景クラス: w1
- 事前確率: P(w0)=0.9, P(w1)=0.1

背景尤度は画素ごとのヒストグラムから与えます。

- P(I|w0) = hist[x,y,I(x,y)] / hist_sum

前景尤度は一様分布で近似します。

- P(I|w1) = 1/255

観測 I の確率は以下です。

- P(I) = P(w0)P(I|w0) + P(w1)P(I|w1)

事後確率はベイズの定理で求めます。

- P(wk|I) = P(I|wk)P(wk)/P(I), k in {0,1}

判定条件は次の通りです。

- 前景 if P(w1|I) > P(w0|I)

この条件を満たす画素を前景として扱い、出力画像では赤色でマーキングします。

## 補足

### Dockerfile の使い分け

- **Dockerfile.dev**  
  開発環境用。COPY は requirements.txt のみ。ソースは共有マウント（volumes）で提供される。  
  ホスト側ファイルの変更がコンテナへ即座に反映される。

- **Dockerfile.prod**  
  本番実行用。COPY . /workspace ですべてのソースをイメージ内に焼き込む。  
  再現性を重視し、ホストファイルへの依存がない。
