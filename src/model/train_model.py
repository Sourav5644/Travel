import sys
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.logger import logging

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """
    Train a RandomForestClassifier with parameters from params.yaml.
    """
    try:
        logging.info('Training RandomForestClassifier with parameters...')
        
        params = load_params(params_path='params.yaml')
        n_estimators = params['model_params']['n_estimators']
        max_features = params['model_params']['max_features']
        criterion = params['model_params']['criterion']

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info('Model training completed.')
        return model

    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        logging.info("Starting model training process...")

        train_data = load_data('./data/interim/train_processed.csv')  # correct path
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        
        save_model(clf, './models/model.pkl')

        logging.info("Model training and saving pipeline completed.")
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
