import pandas as pd
import pickle
import json
import shap
import logging
import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
from typing import Optional
from starlette.status import HTTP_403_FORBIDDEN

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SÉCURITÉ (C19) ---
API_KEY_NAME = "access_token"
API_KEY = os.getenv("API_KEY", "secret-token-12345")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Accès refusé : Clé API invalide ou manquante."
        )

app = FastAPI(
    title="API Scoring Crédit",
    description="API de prédiction du risque de défaut de paiement pour les conseillers clientèle."
)

# Chargement du Pipeline et des artefacts au démarrage
pipeline = None
artifacts = None
explainer = None
model = None # Pour SHAP

try:
    logger.info("Chargement du pipeline et des artefacts...")
    
    # Chargement du Pipeline complet (Imputer -> Scaler -> Model)
    with open("pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
        
    # Extraction du modèle interne pour SHAP
    model = pipeline.named_steps['model']
    
    with open("preprocessing_artifacts.json", "r") as f:
        artifacts = json.load(f)
        
    # Initialisation de l'explainer SHAP sur le modèle XGBoost interne
    # Note: SHAP a besoin des données transformées, c'est la partie délicate
    # On utilisera un TreeExplainer simple
    explainer = shap.TreeExplainer(model)
    
    logger.info("Pipeline et artefacts chargés avec succès.")
except FileNotFoundError as e:
    logger.error(f"ERREUR CRITIQUE : {e}")
except Exception as e:
    logger.error(f"ERREUR inattendue lors du chargement : {e}")

class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., description="Revenus annuels totaux", gt=0)
    AMT_CREDIT: float = Field(..., description="Montant du crédit demandé", gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, description="Montant de l'annuité du crédit")
    CNT_CHILDREN: int = Field(0, description="Nombre d'enfants", ge=0)
    DAYS_BIRTH: int = Field(..., description="Âge en jours (négatif)", le=0)
    DAYS_EMPLOYED: float = Field(..., description="Ancienneté emploi en jours (négatif) ou 365243")

@app.get("/")
def read_root():
    return {"status": "L'API est en ligne", "pipeline_loaded": pipeline is not None}

@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")
    return {"status": "ok"}

@app.post("/predict")
def predict(client: ClientData, api_key: APIKey = Depends(get_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Le pipeline n'est pas chargé.")

    try:
        # 1. Création DataFrame
        input_data = client.dict()
        df = pd.DataFrame([input_data])
        
        # 1.bis Correction anomalies (obligatoire avant Pipeline)
        if 'DAYS_EMPLOYED' in df.columns:
             df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, float('nan'))

        # 2. Prédiction via Pipeline (Gère Imputation + Scaling + Pred)
        # On s'assure d'avoir les colonnes dans le bon ordre
        cols = artifacts.get('final_columns', df.columns.tolist())
        # Ajout colonnes manquantes si besoin (sécurité)
        for c in cols:
            if c not in df.columns:
                df[c] = float('nan')
        df = df[cols]
        
        probability = float(pipeline.predict_proba(df)[:, 1][0])
        
        # Seuil
        threshold = artifacts.get('threshold', 0.5)
        prediction = int(probability > threshold)
        
        # 3. Explicabilité (SHAP)
        # Pour SHAP, il faut passer les données transformées (imputées/scalées)
        # On utilise les étapes du pipeline pour transformer
        preprocessor = Pipeline(pipeline.steps[:-1]) # Tout sauf le modèle
        data_transformed = preprocessor.transform(df)
        
        shap_values = explainer.shap_values(data_transformed)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0] 
        else:
            vals = shap_values[0]
            
        feature_importance = {}
        for i, col in enumerate(cols):
            feature_importance[col] = float(vals[i])

        return {
            "prediction": prediction,
            "probability": probability,
            "threshold": threshold,
            "risk_label": "Risqué" if prediction == 1 else "Sûr",
            "shap_values": feature_importance
        }

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))
