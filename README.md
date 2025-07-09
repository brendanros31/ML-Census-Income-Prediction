# ML-Census-Income-Prediction

This project uses a U.S. Census dataset to predict whether a person's income exceeds $50K/year based on demographic and employment-related features.

Accurate income classification is a valuable task in domains like targeted marketing, financial risk modeling, and public policy research.

We perform **exploratory data analysis (EDA)**, **data cleaning**, **feature engineering**, **data preprocessing**, and apply **multiple machine learning classification models** to build and evaluate a predictive system. 

The goal is to determine which features most influence high-income prediction and which model performs best.

The project evaluates and compares model performance using industry-standard classification metrics, providing insights into both model behavior and the influence of various socio-economic factors on income level.

---

## Dataset Overview

The dataset includes the following columns:

| Feature | Description |
|---------|-------------|
| `age` | Age of the individual |
| `workclass` | Type of employment (e.g., Private, Self-emp, Govt) |
| `education` | Highest level of education completed |
| `education-num` | Numeric representation of education |
| `marital-status` | Marital status |
| `occupation` | Job type |
| `relationship` | Relationship status in the household |
| `race` | Race of the individual |
| `sex` | Gender |
| `capital-gain` | Income from capital gains |
| `capital-loss` | Capital loss |
| `hours-per-week` | Average hours worked per week |
| `native-country` | Country of origin |
| `annual_income` | Target variable: `>50K` or `<=50K` |

---

## Project Workflow

### 1. 🧼 Exploratory Data Analysis (EDA)
- Checked class imbalance in income (`>50K` vs `<=50K`)
- Analyzed distributions of age, education, hours-per-week, etc.
- Visualized relationships between features and income levels
- Assessed correlation and feature importance

### 2. 🧽 Data Cleaning
- **Remapped** categories within key features (e.g., grouped similar education levels or employment classes)
- **Removed redundant values** and consolidated missing/unclear labels using majority values.
- **Re-evaluated impact** of cleaned categorical groups on income prediction
- Saved the cleaned dataset to `data/clean/census-income_clean` for future modeling

### 3. ⚙️ Data Preprocessing
- Used Outlier detection to better understand the Feature ranges
- Applied Z-score to figure how the outliers should be handled
- Reduced Skewness of the data
- Encoded categorical variables using label encoding

### 4. 🤖 Model Training
Trained the following classification models:

- **Logistic Regression**
- **Random Forest Classifier**

### 5. 📊 Model Evaluation

Each model was evaluated using:

| Metric | Description |
|--------|-------------|
| `Accuracy` | Overall correctness |
| `Precision` | TP / (TP + FP) |
| `Recall` | TP / (TP + FN) |
| `F1 Score` | Harmonic mean of precision and recall |
| `Macro Avg` | Unweighted average of metrics across classes |
| `Weighted Avg` | Metric average weighted by class frequencies |
| `Support` | Number of actual examples per class |
| `ROC` | Receiver Operating Characteristic

A **ROC Curve** shows: 
- The trade-off between the true positive rate and false positive rate across different thresholds. 
- Commonly summarized using the AUC (Area Under the Curve), where higher values indicate better classifier performa

---

## How to Run
```bash
git clone https://github.com/brendanros31/ML-Diabetes-Predictive-Analysis.git

cd ML-Diabetes-Predictive-Analysis

pip install -r requirements.txt
python main.ipynb
```

---

## Project Structure
```
data/
    raw/
        census-income.csv
    clean/
        census-income_clean.csv

src/
    data_loader.py
    evaluate.py
    features.py
    model.py
    utils.py

EDA.ipynb
main.ipynb
config/config.yaml
```