import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(input_path, output_path):
    df = pd.read_csv(input_path)

    # Summary statistics
    summary = df.describe()
    summary.to_csv(f'{output_path}/summary_statistics.csv')
    
    # Visualizations
    sns.pairplot(df)
    plt.savefig(f'{output_path}/pairplot.png')

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.savefig(f'{output_path}/correlation_matrix.png')

    print("EDA outputs saved.")

perform_eda('outputs/cleaned_data.csv', 'outputs/eda')
