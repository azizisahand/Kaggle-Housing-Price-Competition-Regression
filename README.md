# Kaggle Housing Price Competition: Regression

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
* **Advanced Modeling: Stacking**
    * To achieve the best predictive score, a `StackingRegressor` was implemented.
    * Multiple base models (e.g., Ridge, Lasso, XGBoost, RandomForestRegressor) were trained.
    * A final meta-model (e.g., a Linear Regression) was trained on the predictions of the base models to produce the final price estimate.
* **Final Model:** The tuned Stacking Regressor was used to generate the final `submission.csv`.

## How to Run

1.  Ensure you have `pandas` and `scikit-learn` installed (and any other libraries for your base models, like `xgboost` or `lightgbm`).
2.  Place the `train.csv` and `test.csv` files in the correct directory (as specified in the notebook).
3.  Run the notebook from top to bottom.
4.  The final predictions will be saved as `submission.csv`.

---

## 🤝 Contact

For any questions or feedback, feel free to reach out:

* **GitHub:** [@zehando](https://github.com/zehando)
* **LinkedIn:** [Sahand Azizi](https://www.linkedin.com/in/sahandazizi/)
* **Email:** azizisahand@gmail.com
