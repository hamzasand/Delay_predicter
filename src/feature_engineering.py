# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

class FeatureEngineer:
    """Creates features for model training"""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from dates"""
        # Start date features
        df['planned_start_month'] = df['planned_start_date'].dt.month
        df['planned_start_quarter'] = df['planned_start_date'].dt.quarter
        df['planned_start_day_of_week'] = df['planned_start_date'].dt.dayofweek
        df['planned_start_week_of_year'] = df['planned_start_date'].dt.isocalendar().week
        
        # Is weekend start
        df['is_weekend_start'] = df['planned_start_day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to task duration"""
        # Already have planned_duration_days from preprocessing
        
        # Duration bins
        df['duration_category'] = pd.cut(
            df['planned_duration_days'],
            bins=[0, 3, 7, 14, 30, np.inf],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )
        
        return df
    
    def create_task_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features indicating task complexity"""
        # Task sequence (assuming task_id represents order)
        df['task_sequence'] = df.groupby('project_id')['task_id'].rank()
        
        # Project size (total tasks per project)
        project_sizes = df.groupby('project_id').size()
        df['project_total_tasks'] = df['project_id'].map(project_sizes)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = ['crew_name', 'region', 'task_label', 'duration_category']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # Crew + Region interaction
        df['crew_region_combo'] = df['crew_name'].astype(str) + '_' + df['region'].astype(str)
        
        # Duration * Month interaction (seasonal complexity)
        df['duration_month_interaction'] = (
            df['planned_duration_days'] * df['planned_start_month']
        )
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Define final feature set for modeling"""
        self.feature_columns = [
            'planned_duration_days',
            'planned_start_month',
            'planned_start_quarter',
            'planned_start_day_of_week',
            'planned_start_week_of_year',
            'is_weekend_start',
            'crew_name_encoded',
            'region_encoded',
            'task_label_encoded',
            'duration_category_encoded',
            'task_sequence',
            'project_total_tasks',
            'duration_month_interaction'
        ]
        
        return self.feature_columns
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Run full feature engineering pipeline"""
        df = self.create_temporal_features(df)
        df = self.create_duration_features(df)
        df = self.create_task_complexity_features(df)
        df = self.encode_categorical_features(df, fit=fit)
        df = self.create_interaction_features(df)
        
        self.select_features(df)
        
        print(f"\nFeature engineering complete. Created {len(self.feature_columns)} features")
        
        return df