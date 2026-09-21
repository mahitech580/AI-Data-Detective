# 🕵️ AI Data Detective

### Automated Dataset Intelligence & Analysis System

**AI Data Detective** is a Python-based data intelligence system that automates the process of exploring, profiling, validating, analyzing, and modeling structured CSV datasets.

The system combines **data profiling, data-quality analysis, statistical analysis, visualization, correlation analysis, and machine learning** into a single interactive workflow. It helps users quickly understand a dataset, identify potential data issues, discover meaningful relationships, and evaluate machine-learning models.

🌐 **Live Project:**
https://mahitech580.github.io/AI-Data-Detective/

---

## 📌 Overview

Analyzing a new dataset often requires several repetitive steps, including checking its structure, handling missing values, detecting duplicates and outliers, calculating statistics, visualizing relationships, and testing machine-learning models.

**AI Data Detective** automates these steps through a unified Python application.

Given a CSV dataset, the system can generate:

* Dataset summaries
* Data-quality information
* Missing-value analysis
* Duplicate detection
* Outlier analysis
* Descriptive statistics
* Correlation analysis
* Automatic visualizations
* Classification model evaluation
* Regression model evaluation
* Model comparison
* Automated analytical reports

The project is designed to make **exploratory data analysis and introductory machine-learning experimentation faster and more systematic**.

---

## 🚀 Key Features

### 📂 Dataset Loading

* Load built-in datasets
* Load custom CSV files
* Automatically inspect dataset structure
* Identify numerical and categorical columns

### 🔎 Automated Dataset Profiling

* Dataset dimensions
* Column information
* Data types
* Unique-value analysis
* Numerical and categorical feature summaries

### 🧹 Data Quality Analysis

* Missing-value detection
* Duplicate-row detection
* Data-type inspection
* Basic dataset-quality checks

### 📊 Statistical Analysis

* Descriptive statistics
* Central tendency measures
* Distribution-related analysis
* Numerical feature summaries

### 📈 Outlier Detection

* Identify potential outliers in numerical columns
* Support exploratory analysis of unusual observations

### 🔗 Correlation Analysis

* Calculate feature correlations
* Identify relationships between numerical variables
* Generate correlation visualizations

### 📉 Automatic Visualization

Generate visualizations for dataset exploration, including:

* Distribution plots
* Histograms
* Box plots
* Correlation heatmaps
* Feature relationships
* Model-related visualizations

### 🤖 Machine Learning

#### Classification

The system supports:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

#### Regression

The system supports:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

### 🏆 Model Comparison

Evaluate multiple machine-learning models and compare their performance using appropriate evaluation metrics.

### 📄 Automated Reporting

Generate structured analytical outputs summarizing:

* Dataset characteristics
* Data-quality findings
* Statistical insights
* Correlations
* Visualizations
* Machine-learning results

### 🖥️ Interactive Terminal Interface

The project provides a terminal-based workflow for interacting with datasets and selecting different analysis operations.

---

## 🧠 Machine Learning Workflow

The machine-learning component follows a general workflow:

```text
Dataset
   ↓
Data Inspection
   ↓
Feature Selection
   ↓
Data Preparation
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
```

The system uses **real Scikit-learn implementations** rather than manually simulated model results.

---

## 🛠️ Tech Stack

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| **Python 3.11+** | Core programming language            |
| **Pandas**       | Data manipulation and analysis       |
| **NumPy**        | Numerical computation                |
| **SciPy**        | Statistical and scientific computing |
| **Scikit-learn** | Machine learning                     |
| **Matplotlib**   | Data visualization                   |
| **Seaborn**      | Statistical visualization            |
| **Rich**         | Interactive terminal interface       |

---

## 🏗️ Project Architecture

```text
AI Data Detective
│
├── Dataset Input
│   └── CSV Files
│
├── Data Profiling
│   ├── Shape
│   ├── Columns
│   ├── Data Types
│   └── Unique Values
│
├── Data Quality
│   ├── Missing Values
│   ├── Duplicates
│   └── Outliers
│
├── Data Analysis
│   ├── Statistics
│   ├── Correlation
│   └── Feature Analysis
│
├── Visualization
│   ├── Distributions
│   ├── Box Plots
│   └── Correlation Heatmaps
│
├── Machine Learning
│   ├── Classification
│   ├── Regression
│   └── Model Comparison
│
└── Reporting
    └── Automated Analysis Results
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mahitech580/AI-Data-Detective.git
```

### 2. Navigate to the Project

```bash
cd AI-Data-Detective
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the application using Python:

```bash
python main.py
```

The terminal interface allows you to select datasets and perform different analysis operations.

> The exact entry-point filename may vary depending on the current repository structure.

---

## 📊 Example Analysis Workflow

A typical workflow looks like this:

```text
Load CSV Dataset
       ↓
Profile Dataset
       ↓
Check Data Quality
       ↓
Analyze Missing Values
       ↓
Detect Duplicates
       ↓
Analyze Outliers
       ↓
Generate Statistics
       ↓
Analyze Correlations
       ↓
Generate Visualizations
       ↓
Run ML Models
       ↓
Compare Models
       ↓
Generate Report
```

---

## 🎯 Use Cases

AI Data Detective can be used for:

* Exploratory Data Analysis (EDA)
* Dataset quality inspection
* Data-science learning
* Machine-learning experimentation
* Feature exploration
* Statistical analysis
* Academic and portfolio projects
* Initial analysis of newly received CSV datasets

---

## 🔬 Why This Project?

The project was built to reduce repetitive exploratory-analysis work by bringing common dataset-analysis tasks into a single workflow.

Instead of manually performing each step independently, users can use **AI Data Detective** to systematically inspect and analyze a dataset before moving into deeper data-science or machine-learning workflows.

---

## 📈 Future Enhancements

Potential improvements include:

* Support for Excel and JSON datasets
* Interactive web-based dashboard
* Advanced feature engineering
* Automated data preprocessing
* Hyperparameter tuning
* Explainable AI and feature importance
* Additional statistical tests
* Interactive Plotly visualizations
* More machine-learning algorithms
* Exportable HTML/PDF reports
* Automated dataset recommendations
* Support for larger datasets

---

## 📁 Dataset Support

The project primarily works with **structured CSV datasets**.

Example:

```text
dataset.csv
```

The system can inspect the dataset and automatically perform the supported profiling and analysis operations.

---

## 📋 Requirements

Recommended environment:

```text
Python 3.11+
Pandas
NumPy
SciPy
Scikit-learn
Matplotlib
Seaborn
Rich
```

---

## ⚠️ Limitations

AI Data Detective is primarily an **automated exploratory-analysis and machine-learning experimentation tool**.

Model performance depends on factors such as:

* Dataset quality
* Feature selection
* Target-variable quality
* Data preprocessing
* Dataset size
* Class distribution
* Model configuration

The generated results should therefore be interpreted as analytical outputs rather than guarantees of real-world model performance.

---

## 🌐 Project Links

**Live Project:**
https://mahitech580.github.io/AI-Data-Detective/

**GitHub Repository:**
https://github.com/mahitech580/AI-Data-Detective

---

## 👨‍💻 Author

**Mahendra Sai Kondaveeti**

Computer Science Engineering Graduate
Python | SQL | Data Analysis | Machine Learning | Full-Stack Development

GitHub:
https://github.com/mahitech580

---

## ⭐ Support

If you find **AI Data Detective** useful for learning, experimentation, or dataset analysis, consider giving the repository a ⭐ on GitHub.

---
