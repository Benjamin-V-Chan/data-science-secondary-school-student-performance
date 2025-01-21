# data-science-secondary-school-student-performance

## Project Overview
This project analyzes and predicts student performance based on social, gender, and academic data. Using exploratory data analysis (EDA) and machine learning models, the project aims to uncover patterns and predict final grades (G3) of students in secondary school math and Portuguese courses.

Key functionalities include:
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Training and evaluating regression models to predict student grades
- Visualizing and interpreting model results

## Folder Structure
```
project/
├── data/                    # Contains the raw dataset(s)
│   └── student-mat.csv      # Dataset for math course students
├── scripts/                 # Python scripts for data processing, analysis, and modeling
│   ├── 01_data_loading_and_cleaning.py
│   ├── 02_exploratory_data_analysis.py
│   ├── 03_model_training.py
│   └── 04_results_analysis_and_visualization.py
├── outputs/                 # Stores all outputs of the project
│   ├── eda/                 # Visualizations and summary statistics from EDA
│   ├── models/              # Trained models and their evaluation metrics
│   └── results/             # Results of the analysis and visualizations
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── .gitignore               # Files to be ignored in version control
```

## Usage
1. **Clone the repository** and navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the scripts** in the following order:
   - **Data Cleaning**: Run `01_data_loading_and_cleaning.py` to clean the dataset.
     ```bash
     python scripts/01_data_loading_and_cleaning.py
     ```
   - **Exploratory Data Analysis**: Perform EDA with `02_exploratory_data_analysis.py`.
     ```bash
     python scripts/02_exploratory_data_analysis.py
     ```
   - **Model Training**: Train regression models using `03_model_training.py`.
     ```bash
     python scripts/03_model_training.py
     ```
   - **Results Analysis**: Analyze and visualize model results with `04_results_analysis_and_visualization.py`.
     ```bash
     python scripts/04_results_analysis_and_visualization.py
     ```

Outputs will be stored in the `outputs/` folder.

## Requirements
- Python 3.8 or later
- Required Python libraries (install with `pip install -r requirements.txt`):
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

## Acknowledgments
- **Dataset Name**: Student Alcohol Consumption
- **Dataset Author**: UCI Machine Learning AND Dmitrii Batogov
- **Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
