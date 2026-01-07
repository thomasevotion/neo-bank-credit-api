import pandas as pd
import pickle
import json
import shap
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Scoring Crédit",
    description="API de prédiction du risque de défaut de paiement pour les conseillers clientèle."
)

# Chargement du modèle et des artefacts au démarrage
model = None
artifacts = None
explainer = None

try:
    logger.info("Chargement du modèle et des artefacts...")
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("preprocessing_artifacts.json", "r") as f:
        artifacts = json.load(f)
        
    # Initialisation de l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    logger.info("Modèle et artefacts chargés avec succès.")
except FileNotFoundError as e:
    logger.error(f"ERREUR CRITIQUE : {e}")
except Exception as e:
    logger.error(f"ERREUR inattendue lors du chargement : {e}")

# Définition du format des données attendues (Contrat d'interface)
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., description="Revenus annuels totaux", gt=0)
    AMT_CREDIT: float = Field(..., description="Montant du crédit demandé", gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, description="Montant de l'annuité du crédit")
    CNT_CHILDREN: int = Field(0, description="Nombre d'enfants", ge=0)
    DAYS_BIRTH: int = Field(..., description="Âge en jours (négatif)", le=0)
    DAYS_EMPLOYED: float = Field(..., description="Ancienneté emploi en jours (négatif) ou 365243")

@app.get("/")
def read_root():
    return {"status": "L'API est en ligne", "model_loaded": model is not None}

@app.get("/health")
def health_check():
    """Endpoint de monitoring pour vérifier la santé de l'API."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok"}

@app.post("/predict")
def predict(client: ClientData):
    if model is None or artifacts is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    try:
        logger.info("Réception d'une demande de prédiction.")
        
        # 1. Conversion des données reçues en DataFrame
        input_data = client.dict()
        df = pd.DataFrame([input_data])
        
        # 1.bis Correction des anomalies
        if 'DAYS_EMPLOYED' in df.columns:
             df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, float('nan'))

        # 2. Prétraitement
        for col, median_val in artifacts['medians'].items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
                if df[col].isnull().any():
                    df[col] = df[col].fillna(median_val)
        
        # 3. Alignement des colonnes
        df_final = pd.DataFrame(columns=artifacts['final_columns'])
        for col in artifacts['final_columns']:
            if col in df.columns:
                df_final[col] = df[col]
            else:
                df_final[col] = 0
        df_final = df_final.fillna(0)
        
        # 4. Prédiction
        probability = float(model.predict_proba(df_final)[:, 1][0])
        
        # Seuil de décision métier (chargé depuis artifacts)
        threshold = artifacts.get('threshold', 0.5)
        prediction = int(probability > threshold)
        
        logger.info(f"Prédiction effectuée : Proba={probability:.2f}, Seuil={threshold:.2f}")

        # 5. Explicabilité (SHAP)
        shap_values = explainer.shap_values(df_final)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0] 
        else:
            vals = shap_values[0]
            
        feature_importance = {}
        for i, col in enumerate(artifacts['final_columns']):
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
