import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import os
# --- INJEKSI KREDENSIAL DAGSHUB ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "ridhooabdee"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c88d510229a9efea57334099a907d03a5ba72daa"

DAGSHUB_TRACKING_URI = "https://dagshub.com/ridhooabdee/Eksperimen_SML_RidhoAbdeeNugraha.mlflow" 
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
mlflow.set_experiment("Basic_Model_Kelayakan_Beasiswa")

def main():
    # Aktifkan Autolog (Syarat Basic)
    mlflow.sklearn.autolog()
    
    data_path = 'preprocessing/dataset-kelayakan-beasiswa_preprocessing/dataset_kelayakan_beasiswa_clean.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Status_Kelayakan'])
    y = df['Status_Kelayakan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), num_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
    
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42))
    ])
    
    with mlflow.start_run(run_name="RF_Basic_Autolog"):
        rf_pipeline.fit(X_train, y_train)
        print("Selesai melatih model Basic!")

if __name__ == "__main__":
    main()