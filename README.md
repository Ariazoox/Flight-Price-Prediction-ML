# Flight Price Prediction - Machine Learning Project

A comprehensive machine learning project that predicts airline ticket prices using various regression models and data analysis techniques.

## 📋 Overview

This project aims to build predictive models capable of estimating airline ticket prices based on structured flight data. The project includes exploratory data analysis, unsupervised learning techniques, data preprocessing, and baseline model implementations using both linear and ensemble methods.

## 🎯 Project Objectives

- Perform comprehensive exploratory data analysis (EDA) on flight pricing data
- Conduct unsupervised exploration using clustering techniques
- Implement data preprocessing and feature engineering pipelines
- Build and evaluate baseline models (Linear Regression and Random Forest)
- Compare model performance using cross-validation

## 📊 Dataset

The dataset contains **300,153 flight records** with the following features:

- **airline**: Airline company name
- **flight**: Flight number
- **source_city**: Departure city
- **departure_time**: Time of departure
- **stops**: Number of stops
- **arrival_time**: Time of arrival
- **destination_city**: Arrival city
- **class**: Flight class (Economy, Business, etc.)
- **duration**: Flight duration in hours
- **days_left**: Days until departure
- **price**: Target variable (ticket price in USD)

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models and preprocessing
  - Linear Regression
  - Random Forest Regressor
  - KMeans Clustering
  - StandardScaler, OrdinalEncoder
  - Pipeline, ColumnTransformer

## 📁 Project Structure

```
Flight-Price-Prediction-ML/
│
├── Clean_Dataset.csv          # Cleaned flight price dataset
├── Projet.ipynb               # Initial project notebook
├── ProjetV2.ipynb             # Main project notebook (Phase 1)
└── README.md                  # Project documentation
```

## 🔬 Methodology

### 1. Data Exploration & Analysis
- Statistical summary and data profiling
- Distribution analysis of numerical features
- Categorical feature analysis
- Correlation analysis
- Visualization of relationships between features and target

### 2. Unsupervised Exploration
- KMeans clustering analysis
- Cluster visualization (duration vs price)
- Pattern discovery in flight pricing

### 3. Data Preprocessing
- **Feature Categorization**:
  - Numerical features: `duration`, `days_left`
  - Ordinal features: `class`, `departure_time`, `arrival_time`
  - Categorical features: `airline`, `flight`, `source_city`, `stops`, `destination_city`

- **Preprocessing Pipeline**:
  - Missing value imputation (median for numerical, most frequent for categorical)
  - Feature scaling (StandardScaler for numerical features)
  - Encoding (OrdinalEncoder for categorical and ordinal features)

### 4. Model Training & Evaluation

#### Baseline Models:
1. **Linear Regression**
   - Train Score: ~0.905
   - Validation Score: ~0.905
   - Test Score: ~0.905

2. **Random Forest Regressor**
   - Test Score: ~0.986

### 5. Model Evaluation
- 5-fold cross-validation
- Train/validation/test split (80/20)
- Performance metrics: R² score

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Flight-Price-Prediction-ML.git
cd Flight-Price-Prediction-ML
```

2. Open the notebook:
```bash
jupyter notebook ProjetV2.ipynb
```

Or use Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ariazoox/Flight-Price-Prediction-ML/blob/main/Projet.ipynb)

3. The dataset is automatically loaded from the GitHub repository.

## 📈 Results

The Random Forest model achieved superior performance with an R² score of approximately **0.986**, significantly outperforming the Linear Regression baseline model. This indicates that non-linear relationships and feature interactions play a crucial role in flight price prediction.

## 🔮 Future Improvements

- [ ] Feature engineering (e.g., time-based features, route combinations)
- [ ] Hyperparameter tuning for Random Forest
- [ ] Advanced ensemble methods (XGBoost, LightGBM, Gradient Boosting)
- [ ] Deep learning models (Neural Networks)
- [ ] Model interpretability analysis (SHAP values, feature importance)
- [ ] Deployment as a web application or API

## 📝 Project Phase

This project represents **Phase 1** of the machine learning pipeline, focusing on:
- ✅ Data Exploratory Analysis & Unsupervised Exploration
- ✅ Data preprocessing, preparation & train-val-test splits
- ✅ Baseline results with basic Linear & Ensemble Models

## 👥 Contributors

- Project maintained by the development team

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset sourced from Kaggle (Flight Price Prediction)
- Built as part of a machine learning course project

---

**Note**: This is an ongoing project. Contributions and suggestions are welcome!
