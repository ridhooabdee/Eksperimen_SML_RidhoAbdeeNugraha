import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import os

# --- INJEKSI KREDENSIAL DAGSHUB ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "ridhooabdee"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c88d510229a9efea57334099a907d03a5ba72daa"
# 1. SETUP DAGSHUB & MLFLOW
# PENTING: Ganti string di bawah dengan URL MLflow dari DagsHub Anda
DAGSHUB_TRACKING_URI = "https://dagshub.com/ridhooabdee/Eksperimen_SML_RidhoAbdeeNugraha.mlflow" 
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
mlflow.set_experiment("Tuning_Model_Kelayakan_Beasiswa")

def main():
    print("=== Memulai Proses Training dengan Pipeline & MLflow ===")
    
    # 2. LOAD DATA
    # Path sudah disesuaikan dengan folder dataset Anda
    data_path = 'preprocessing/dataset-kelayakan-beasiswa_preprocessing/dataset_kelayakan_beasiswa_clean.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Status_Kelayakan'])
    y = df['Status_Kelayakan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. PIPELINE PREPROCESSING
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ])
    
    # 4. PIPELINE MODEL & HYPERPARAMETER GRID
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    # Parameter untuk di-tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [4, 6],
        'classifier__min_samples_split': [10, 20]
    }
    
    grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    # 5. MULAI MLFLOW RUN (SYARAT ADVANCE - MANUAL LOGGING)
    with mlflow.start_run(run_name="RF_Pipeline_Tuning"):
        print("Sedang melatih model dan mencari parameter terbaik...")
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        print(f"Parameter terbaik ditemukan: {grid_search.best_params_}")
        
        # Prediksi ke Test Set
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)
        
        # Hitung Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred_proba)
        
        # Log Parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("log_loss", ll)
        
        # 6. ARTIFAK GAMBAR (SYARAT ADVANCE)
        os.makedirs("artifacts", exist_ok=True)
        
        # Artifak 1: Confusion Matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Prediksi Kelayakan")
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Artifak 2: Feature Importance (Mengambil dari dalam pipeline)
        plt.figure(figsize=(10,6))
        # Mendapatkan nama fitur setelah one-hot encoding
        cat_encoder = best_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(cat_cols)
        all_features = num_cols + list(cat_features)
        
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        feat_importances = pd.Series(importances, index=all_features)
        feat_importances.nlargest(10).plot(kind='barh', color='teal').invert_yaxis()
        plt.title("10 Fitur Paling Berpengaruh")
        
        feat_path = "artifacts/feature_importance.png"
        plt.savefig(feat_path)
        mlflow.log_artifact(feat_path)
        plt.close()
        
        # Menyimpan Model
        mlflow.sklearn.log_model(best_pipeline, "model_random_forest")
        
        print("\nSelesai! Pipeline Model, Metrik, dan Artefak telah berhasil di-push ke DagsHub.")

if __name__ == "__main__":
    main()