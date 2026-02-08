#!/bin/bash

# 心不全転帰先予測システム 起動スクリプト

cd "$(dirname "$0")"

# 仮想環境をアクティベート
source venv/bin/activate

echo "=================================="
echo "心不全転帰先予測システム"
echo "=================================="
echo ""
echo "APIサーバーを起動中..."
echo "API: http://localhost:8000"
echo "フロントエンド: frontend/index.html をブラウザで開いてください"
echo ""
echo "終了するには Ctrl+C を押してください"
echo ""

# FastAPIサーバーを起動
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
