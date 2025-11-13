# Kaggle Housing Price Competition: Regression

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-336791?style=for-the-badge&logoColor=white)

This project is a solution for the Kaggle competition to predict house sale prices (a regression task) based on the Ames Housing dataset.

## Project Overview

This solution uses a complete `scikit-learn` pipeline and advanced stacking techniques to process the data, train a model, and predict the final `SalePrice`.

The workflow includes:

* **Feature Engineering:** Creating 6 new features, including `Total_SF`, `Total_Bathrooms`, and `House_Age`, to improve predictive power.
* **Preprocessing Pipeline:** A robust pipeline was built to:
    * Impute missing numerical and categorical data.
    * Apply `OrdinalEncoder` to 22 features with custom, defined orderings.
    * Apply `OneHotEncoder` to the remaining 22 nominal features.
    * Scale all features using `StandardScaler`.
* **Model Tuning:** `GridSearchCV` was used to tune and compare four different models:
    1.  `DecisionTreeClassifier`
    2.  `KNeighborsClassifier`
    3.  `RandomForestClassifier`
    4.  `XGBClassifier`
* **Final Model:** The tuned **XGBClassifier** was identified as the best-performing model and was used to generate the final submission file.

## How to Run

1.  Ensure you have `pandas` and `scikit-learn` installed (and any other libraries for your base models, like `xgboost` or `lightgbm`).
2.  Place the `labeled.csv` (train data) and `unlabeled.csv` (test data) files in the correct directory (as specified in the notebook).
3.  Run the notebook from top to bottom.
4.  The final predictions will be saved as `submission.csv`.

---

## 🤝 Contact

For any questions or feedback, feel free to reach out:

* **GitHub:** [@zehando](https://github.com/zehando)
* **LinkedIn:** [Sahand Azizi](https://www.linkedin.com/in/sahandazizi/)
* **Email:** azizisahand@gmail.com
