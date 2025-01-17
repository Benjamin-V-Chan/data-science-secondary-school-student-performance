import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_clean_data(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical variables
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

load_and_clean_data('data/student-mat.csv', 'outputs/cleaned_data.csv')