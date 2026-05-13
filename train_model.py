"""
train_model.py
──────────────
Script de entrenamiento del modelo NÍTIDO.
Ejecutar UNA SOLA VEZ antes de lanzar la app:
    python train_model.py

Genera los archivos:
    model_nitido.pkl
    scaler_nitido.pkl
    explainer_nitido.pkl
    feature_cols.pkl
"""

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. Cargar datos
# ─────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv("candidatos_nitido.csv")

FEAT_COLS = [c for c in df.columns if c != "avanza"]
TARGET    = "avanza"

# ─────────────────────────────────────────
# 2. Imputación de nulos (mediana)
# ─────────────────────────────────────────
for col in FEAT_COLS:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# ─────────────────────────────────────────
# 3. Partición 70 / 15 / 15
# ─────────────────────────────────────────
X = df[FEAT_COLS]
y = df[TARGET]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────
# 4. Estandarización
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 5. SMOTE (desbalance leve ~52/48, aplicamos igual)
# ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)
print(f"Después de SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ─────────────────────────────────────────
# 6. Modelo base: Regresión Logística
# ─────────────────────────────────────────
print("Entrenando modelo base (Regresión Logística)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val_s)[:, 1])
print(f"  LR  AUC val: {lr_auc:.4f}")

# ─────────────────────────────────────────
# 7. Modelos complejos
# ─────────────────────────────────────────
print("Entrenando Random Forest...")
rf_grid = {"n_estimators": [100, 200], "max_depth": [6, 10, None]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    rf_grid, scoring="roc_auc", cv=cv, n_jobs=-1
)
rf_gs.fit(X_train_res, y_train_res)
rf_best = rf_gs.best_estimator_
rf_auc  = roc_auc_score(y_val, rf_best.predict_proba(X_val_s)[:, 1])
print(f"  RF  AUC val: {rf_auc:.4f}  params: {rf_gs.best_params_}")

print("Entrenando Gradient Boosting...")
gb_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
gb_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_grid, scoring="roc_auc", cv=cv, n_jobs=-1
)
gb_gs.fit(X_train_res, y_train_res)
gb_best = gb_gs.best_estimator_
gb_auc  = roc_auc_score(y_val, gb_best.predict_proba(X_val_s)[:, 1])
print(f"  GB  AUC val: {gb_auc:.4f}  params: {gb_gs.best_params_}")

# ─────────────────────────────────────────
# 8. Elegir modelo final
# ─────────────────────────────────────────
results = {"LR": (lr, lr_auc), "RF": (rf_best, rf_auc), "GB": (gb_best, gb_auc)}
best_name = max(results, key=lambda k: results[k][1])
best_model = results[best_name][0]
print(f"\nModelo seleccionado: {best_name} (AUC val = {results[best_name][1]:.4f})")

# Evaluación en test
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_s)[:, 1])
print(f"AUC en TEST: {test_auc:.4f}")
print(classification_report(y_test, best_model.predict(X_test_s)))

# ─────────────────────────────────────────
# 9. SHAP Explainer
# ─────────────────────────────────────────
print("Creando SHAP explainer...")
X_train_df = pd.DataFrame(X_train_res, columns=FEAT_COLS)

if best_name in ("RF", "GB"):
    explainer = shap.TreeExplainer(best_model, X_train_df)
else:
    explainer = shap.LinearExplainer(best_model, X_train_df)

# ─────────────────────────────────────────
# 10. Guardar artefactos
# ─────────────────────────────────────────
joblib.dump(best_model, "model_nitido.pkl")
joblib.dump(scaler,     "scaler_nitido.pkl")
joblib.dump(explainer,  "explainer_nitido.pkl")
joblib.dump(FEAT_COLS,  "feature_cols.pkl")

print("\n✅ Artefactos guardados:")
print("   model_nitido.pkl")
print("   scaler_nitido.pkl")
print("   explainer_nitido.pkl")
print("   feature_cols.pkl")
print("\nYa puedes lanzar la app con:  streamlit run app.py")
