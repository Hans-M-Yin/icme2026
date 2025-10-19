#!/bin/bash

# 创建目录结构
MODEL_DIR="./models/gpt2"
mkdir -p $MODEL_DIR

# 下载所有必要文件
wget -P $MODEL_DIR/ https://huggingface.co/datasets/huanngzh/3D-Front/resolve/main/3D-FRONT-TEST-RENDER.tar.gz?download=true
wget -P $MODEL_DIR/ https://huggingface.co/datasets/huanngzh/3D-Front/resolve/main/3D-FRONT-TEST-SCENE.tar.gz?download=true
wget -P $MODEL_DIR/ https://huggingface.co/gpt2/resolve/main/vocab.json
wget -P $MODEL_DIR/ https://huggingface.co/gpt2/resolve/main/tokenizer.json

echo "所有文件已下载到: $MODEL_DIR"