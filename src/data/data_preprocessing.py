import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.logger import logging  # Custom logger

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects top 8 important columns, handles missing values, and encodes categorical variables.
    Returns the preprocessed DataFrame.
    """
    try:
        logging.info(f"Original data shape: {df.shape}")

        # Select top 8 features based on notebook analysis
        selected_cols = [
            'Passport', 'MaritalStatus', 'Age',
            'ProductPitched', 'MonthlyIncome',
            'NumberOfFollowups', 'Designation',
            'PreferredPropertyStar','ProdTaken'
        ]
        df = df[selected_cols]
        logging.info(f"Selected columns: {selected_cols}")

        # Drop missing values
        df = df.dropna()
        logging.info("Dropped rows with missing values")

        # Encode categorical features
        cat_cols = ['MaritalStatus', 'ProductPitched', 'Designation']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logging.info(f"Encoded column: {col}")

        return df

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


def main():
    """
    Loads train/test data, preprocesses it, and saves the results to data/processed/
    """
    try:
        logging.info("Loading raw train and test datasets...")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        # Apply preprocessing
        train_processed = preprocess(train_data)
        test_processed = preprocess(test_data)

        # Save preprocessed data
        output_dir = os.path.join('./data','interim')
        os.makedirs(output_dir, exist_ok=True)

        train_processed.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

        logging.info("Preprocessed data saved successfully.")

    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
