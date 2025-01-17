import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_models(input_path, output_path):
    df = pd.read_csv(input_path)
    X = df.drop('G3', axis=1)
    y = df['G3']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_metrics = {
        'R2': r2_score(y_test, lr_preds),
        'MSE': mean_squared_error(y_test, lr_preds)
    }

    # Train Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = {
        'R2': r2_score(y_test, rf_preds),
        'MSE': mean_squared_error(y_test, rf_preds)
    }

    # Save models and metrics
    joblib.dump(lr, f'{output_path}/linear_regression_model.pkl')
    joblib.dump(rf, f'{output_path}/random_forest_model.pkl')

    pd.DataFrame([lr_metrics, rf_metrics], index=['Linear Regression', 'Random Forest']).to_csv(f'{output_path}/model_metrics.csv')
    print("Models trained and saved.")

train_models('outputs/cleaned_data.csv', 'outputs/models')
