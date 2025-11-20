import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from typing import Tuple, Dict
import os

class ModelTrainer:
    """Trains and evaluates the delay prediction model"""
    
    def __init__(self, feature_columns: list):
        self.feature_columns = feature_columns
        self.model = None
        self.metrics = {}
        
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'delay_days',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and validation sets"""
        
        X = df[self.feature_columns]
        y = df[target_col]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
        """Train XGBoost regression model"""
        
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining XGBoost model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
        
        return self.model
    
    def evaluate_model(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model on validation set"""
        
        y_pred = self.model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print("\n" + "="*50)
        print("VALIDATION METRICS")
        print("="*50)
        print(f"MAE (Mean Absolute Error):  {mae:.2f} days")
        print(f"RMSE (Root Mean Squared Error): {rmse:.2f} days")
        print(f"R² Score: {r2:.4f}")
        print("="*50)
        
        return self.metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from trained model"""
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_model(self, filepath: str = 'models/xgboost_model.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/xgboost_model.pkl'):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model