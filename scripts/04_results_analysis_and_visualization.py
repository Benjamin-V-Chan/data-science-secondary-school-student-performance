import pandas as pd
import matplotlib.pyplot as plt
import joblib

def analyze_results(model_path, input_path, output_path):
    df = pd.read_csv(input_path)
    X = df.drop('G3', axis=1)
    y = df['G3']

    # Load the model
    model = joblib.load(model_path)

    # Predict and visualize
    predictions = model.predict(X)
    plt.scatter(y, predictions)
    plt.xlabel("Actual Grades")
    plt.ylabel("Predicted Grades")
    plt.title("Model Predictions vs Actuals")
    plt.savefig(f'{output_path}/predictions_vs_actuals.png')
    print("Results analysis completed.")

analyze_results('outputs/models/random_forest_model.pkl', 'outputs/cleaned_data.csv', 'outputs/results')
