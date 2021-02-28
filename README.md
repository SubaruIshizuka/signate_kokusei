# 概要

signate 国勢調査からの収入予測
14 / 391 位

## 環境
python: 3.7.2

# 手順

## クローン
```sh
git clone https://github.com/takapy0210/ml_pipeline.git
```

## フォルダ移動
```sh
cd signate_kokusei/code
```

## 特徴量生成
```sh
python create_features.py
```

## 学習
```sh
python kokusei_run.py
```
