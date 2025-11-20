import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

class DataPreprocessor:
    """Handles data loading, cleaning, and basic preprocessing"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def parse_dates(self) -> pd.DataFrame:
        """Convert date columns to datetime"""
        date_columns = [
            'planned_start_date', 'planned_end_date',
            'actual_start_date', 'actual_end_date'
        ]
        
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], format='%Y-%m-%d', errors='coerce')
        
        print(f"Parsed dates. Missing values: {self.df[date_columns].isnull().sum().sum()}")
        return self.df
    
    def compute_durations(self) -> pd.DataFrame:
        """Calculate planned and actual durations"""
        self.df['planned_duration_days'] = (
            self.df['planned_end_date'] - self.df['planned_start_date']
        ).dt.days + 1  # +1 to include both start and end days
        
        self.df['actual_duration_days'] = (
            self.df['actual_end_date'] - self.df['actual_start_date']
        ).dt.days + 1
        
        return self.df
    
    def compute_target(self) -> pd.DataFrame:
        """Calculate delay in days (target variable)"""
        self.df['delay_days'] = (
            self.df['actual_duration_days'] - self.df['planned_duration_days']
        )
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values strategically"""
        # Remove rows with missing critical dates
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[
            'planned_start_date', 'planned_end_date',
            'actual_start_date', 'actual_end_date'
        ])
        print(f"Removed {initial_count - len(self.df)} rows with missing dates")
        
        # Fill missing categorical values
        self.df['crew_name'] = self.df['crew_name'].fillna('unknown')
        self.df['task_label'] = self.df['task_label'].fillna('unknown')
        self.df['region'] = self.df['region'].fillna('unknown')
        
        return self.df
    
    def remove_outliers(self, column: str = 'delay_days', threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        initial_count = len(self.df)
        self.df = self.df[
            (self.df[column] >= lower_bound) & 
            (self.df[column] <= upper_bound)
        ]
        print(f"Removed {initial_count - len(self.df)} outliers from {column}")
        
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """Run full preprocessing pipeline"""
        self.load_data()
        self.parse_dates()
        self.compute_durations()
        self.compute_target()
        self.handle_missing_values()
        self.remove_outliers()
        
        print(f"\nPreprocessing complete. Final dataset: {len(self.df)} rows")
        print(f"Target statistics:\n{self.df['delay_days'].describe()}")
        
        return self.df