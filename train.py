import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
import joblib

def main():
    print("="*60)
    print("CONSTRUCTION TASK DELAY PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # 1. Preprocess data
    print("\n[STEP 1] Data Preprocessing")
    print("-" * 60)
    preprocessor = DataPreprocessor('data/construction_clean1.csv')
    df_clean = preprocessor.preprocess()
    
    # 2. Engineer features
    print("\n[STEP 2] Feature Engineering")
    print("-" * 60)
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df_clean, fit=True)
    
    # 3. Train model
    print("\n[STEP 3] Model Training")
    print("-" * 60)
    trainer = ModelTrainer(feature_engineer.feature_columns)
    X_train, X_val, y_train, y_val = trainer.split_data(df_features)
    trainer.train_model(X_train, y_train)
    
    # 4. Evaluate
    print("\n[STEP 4] Model Evaluation")
    print("-" * 60)
    trainer.evaluate_model(X_val, y_val)
    trainer.get_feature_importance()
    
    # 5. Save artifacts
    print("\n[STEP 5] Saving Artifacts")
    print("-" * 60)
    trainer.save_model('models/xgboost_model.pkl')
    joblib.dump(feature_engineer, 'models/feature_engineer.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()