import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    mlflow.sklearn.autolog()
    
    data_path = 'dataset-kelayakan-beasiswa_preprocessing/dataset_kelayakan_beasiswa_clean.csv'
    df = pd.read_csv(data_path)
    
    X = df[['IPK', 'Skor_Motivasi_Tertulis', 'Skor_Literasi_Finansial', 'Semester']]
    y = df['Status_Kelayakan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Run_Lulus") as run:
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        os.makedirs("folder_pancingan", exist_ok=True)
        joblib.dump(model, "folder_pancingan/model.pkl")
        
        with open("folder_pancingan/conda.yaml", "w") as f:
            f.write("name: submission_env")
        with open("folder_pancingan/MLmodel", "w") as f:
            f.write("flavor: sklearn")
            
        mlflow.log_artifacts("folder_pancingan", artifact_path="model")
        
        print(f"SUKSES! Folder model dijamin ada.")

if __name__ == "__main__":
    main()