"""
心不全患者の転帰先予測API
FastAPIを使用したRESTful API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import json
import os

app = FastAPI(
    title="心不全転帰先予測API",
    description="入院時データから転帰先（自宅・転院・施設）を予測するAPI",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルと設定の読み込み
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
le_diagnosis = joblib.load(os.path.join(models_dir, 'label_encoder_diagnosis.pkl'))

with open(os.path.join(models_dir, 'config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)

# 診断名の逆マッピング（日本語 → 数値）
diagnosis_to_code = {v: k for k, v in config['diagnosis_mapping'].items()}

class PatientData(BaseModel):
    """患者入院時データ（必須: 年齢、性別、診断名のみ。他はオプション）"""
    # 基本情報（必須）
    年齢: int = Field(..., ge=0, le=120, description="年齢")
    性別: int = Field(..., ge=0, le=1, description="性別 (0: 女性, 1: 男性)")
    診断名: str = Field(..., description="診断名 (HFrEF, HFmrEF, HFpEF, 弁膜症性心不全, 心筋梗塞後)")

    # 基本情報（オプション）
    体重: Optional[float] = Field(default=60.0, ge=20, le=200, description="体重 (kg)")
    BMI: Optional[float] = Field(default=22.0, ge=10, le=60, description="BMI")
    介護保険認定: Optional[int] = Field(default=0, ge=0, le=7, description="介護保険認定")
    独居_支援: Optional[int] = Field(default=0, ge=0, le=1, alias="独居・支援", description="独居・支援の有無")

    # 心機能（オプション）
    NYHA: Optional[int] = Field(default=2, ge=1, le=4, description="NYHA分類 (1-4)")
    心不全入院歴: Optional[int] = Field(default=0, ge=0, description="心不全入院歴 (回数)")
    原疾患: Optional[int] = Field(default=0, ge=0, le=1, description="原疾患の有無")

    # 併存疾患（オプション）
    高血圧: Optional[int] = Field(default=0, ge=0, le=1, description="高血圧")
    糖尿病: Optional[int] = Field(default=0, ge=0, le=1, description="糖尿病")
    腎疾患: Optional[int] = Field(default=0, ge=0, le=1, description="腎疾患")
    運動器疾患: Optional[int] = Field(default=0, ge=0, le=1, description="運動器疾患")
    脳血管疾患: Optional[int] = Field(default=0, ge=0, le=1, description="脳血管疾患")

    # 心エコー（オプション）
    EF: Optional[float] = Field(default=50.0, ge=0, le=100, description="EF (%)")
    LAD: Optional[float] = Field(default=40.0, ge=0, le=100, description="LAD (mm)")

    # 血液検査（オプション）
    血清Cre: Optional[float] = Field(default=1.0, ge=0, description="血清Cre (mg/dL)")
    血清Alb: Optional[float] = Field(default=3.5, ge=0, description="血清Alb (g/dL)")
    血清BUN: Optional[float] = Field(default=20.0, ge=0, description="血清BUN (mg/dL)")
    血清eGFR: Optional[float] = Field(default=60.0, ge=0, description="血清eGFR")
    血清CRP: Optional[float] = Field(default=0.3, ge=0, description="血清CRP (mg/dL)")
    血清Hb: Optional[float] = Field(default=12.0, ge=0, description="血清Hb (g/dL)")
    血清Na: Optional[float] = Field(default=140.0, ge=100, le=160, description="血清Na (mEq/L)")
    NT_proBNP: Optional[float] = Field(default=1000.0, ge=0, alias="NT-proBNP", description="NT-proBNP (pg/mL)")

    # 投薬（オプション）
    βブロッカー: Optional[int] = Field(default=0, ge=0, le=1, description="βブロッカー")
    ACE阻害薬: Optional[int] = Field(default=0, ge=0, le=1, description="ACE阻害薬")
    SGLT2阻害薬: Optional[int] = Field(default=0, ge=0, le=1, description="SGLT2阻害薬")
    ARB: Optional[int] = Field(default=0, ge=0, le=1, description="ARB")
    MRA: Optional[int] = Field(default=0, ge=0, le=1, description="MRA")
    ARNI: Optional[int] = Field(default=0, ge=0, le=1, description="ARNI")
    デバイス挿入: Optional[int] = Field(default=0, ge=0, le=1, description="デバイス挿入")

    # 入院時ADL（オプション）
    BI入院: Optional[float] = Field(default=85.0, ge=0, le=100, description="BI入院 (0-100)")
    FIM入院: Optional[float] = Field(default=100.0, ge=18, le=126, description="FIM入院 (18-126)")
    バランス入院: Optional[float] = Field(default=12.0, ge=0, description="バランス入院スコア")
    起立入院: Optional[float] = Field(default=3.0, ge=0, description="起立入院スコア")
    歩行入院: Optional[float] = Field(default=4.0, ge=0, description="歩行入院スコア")
    合計入院: Optional[float] = Field(default=19.0, ge=0, description="合計入院スコア")

    # 身体機能（オプション）
    歩行_4m: Optional[float] = Field(default=5.0, ge=0, alias="4m歩行", description="4m歩行 (秒)")
    通常歩行速度: Optional[float] = Field(default=0.8, ge=0, description="通常歩行速度 (m/s)")
    握力: Optional[float] = Field(default=25.0, ge=0, description="握力 (kg)")

    # 認知機能（オプション）
    認知機能低下: Optional[int] = Field(default=0, ge=0, le=1, description="認知機能低下")
    基本チェックリスト: Optional[int] = Field(default=5, ge=0, le=25, description="基本チェックリスト (0-25)")

    class Config:
        populate_by_name = True

class PredictionResult(BaseModel):
    """予測結果"""
    prediction: str = Field(..., description="予測された転帰先")
    prediction_code: int = Field(..., description="予測コード (0: 自宅, 1: 転院, 2: 施設)")
    probabilities: dict = Field(..., description="各クラスの確率")
    model_accuracy: float = Field(..., description="モデルの精度")

@app.get("/")
def read_root():
    """ルートエンドポイント"""
    return {
        "message": "心不全転帰先予測API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - 転帰先を予測",
            "/diagnosis-options": "GET - 診断名の選択肢",
            "/health": "GET - ヘルスチェック"
        }
    }

@app.get("/health")
def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/diagnosis-options")
def get_diagnosis_options():
    """診断名の選択肢を取得"""
    return {
        "options": list(diagnosis_to_code.keys()),
        "mapping": diagnosis_to_code
    }

@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientData):
    """患者データから転帰先を予測"""
    try:
        # 診断名をエンコード
        if patient.診断名 not in diagnosis_to_code:
            raise HTTPException(
                status_code=400,
                detail=f"無効な診断名です。有効な選択肢: {list(diagnosis_to_code.keys())}"
            )

        diagnosis_encoded = diagnosis_to_code[patient.診断名]

        # 特徴量を配列に変換（順序が重要）
        features = [
            patient.年齢,
            patient.性別,
            patient.体重,
            patient.BMI,
            patient.介護保険認定,
            patient.独居_支援,
            diagnosis_encoded,
            patient.NYHA,
            patient.心不全入院歴,
            patient.原疾患,
            patient.高血圧,
            patient.糖尿病,
            patient.腎疾患,
            patient.運動器疾患,
            patient.脳血管疾患,
            patient.EF,
            patient.LAD,
            patient.血清Cre,
            patient.血清Alb,
            patient.血清BUN,
            patient.血清eGFR,
            patient.血清CRP,
            patient.血清Hb,
            patient.血清Na,
            patient.NT_proBNP,
            patient.βブロッカー,
            patient.ACE阻害薬,
            patient.SGLT2阻害薬,
            patient.ARB,
            patient.MRA,
            patient.ARNI,
            patient.デバイス挿入,
            patient.BI入院,
            patient.FIM入院,
            patient.バランス入院,
            patient.起立入院,
            patient.歩行入院,
            patient.合計入院,
            patient.歩行_4m,
            patient.通常歩行速度,
            patient.握力,
            patient.認知機能低下,
            patient.基本チェックリスト,
        ]

        # スケーリング
        features_scaled = scaler.transform([features])

        # 予測
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # 結果を整形
        target_labels = config['target_labels']

        return PredictionResult(
            prediction=target_labels[str(prediction)],
            prediction_code=int(prediction),
            probabilities={
                target_labels[str(i)]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
            model_accuracy=config['accuracy']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
