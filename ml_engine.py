import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MLEngine:
    CLASSIFIERS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    REGRESSORS = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }

    def available_models(self, task):
        return list(
            self.CLASSIFIERS if task == "classification"
            else self.REGRESSORS
        )

    def train(self, df, target, task, model_name):
        if target not in df.columns:
            raise ValueError("Target column not found.")

        work = df.dropna(subset=[target]).copy()

        X = work.drop(columns=[target])
        y = work[target]

        if len(X) < 10:
            raise ValueError("Dataset is too small for reliable training.")

        if task == "classification" and y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        numeric = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical = [
            col for col in X.columns
            if col not in numeric
        ]

        transformers = []

        if numeric:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ))

        if categorical:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="most_frequent"
                    )),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore"
                    )),
                ]),
                categorical,
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        stratify = None

        if task == "classification":
            if y.value_counts().min() >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )

        registry = (
            self.CLASSIFIERS
            if task == "classification"
            else self.REGRESSORS
        )

        if model_name not in registry:
            raise ValueError("Unknown model.")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", registry[model_name]),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        if task == "classification":
            metrics = {
                "accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        else:
            mse = mean_squared_error(
                y_test,
                predictions
            )

            metrics = {
                "mae": float(
                    mean_absolute_error(
                        y_test,
                        predictions
                    )
                ),
                "rmse": float(np.sqrt(mse)),
                "r2": float(
                    r2_score(
                        y_test,
                        predictions
                    )
                ),
            }

        return {
            "pipeline": pipeline,
            "metrics": metrics,
            "predictions": predictions,
            "X_test": X_test,
            "y_test": y_test,
            "target": target,
            "task": task,
            "model": model_name,
        }

    def compare(self, df, target, task):
        results = []

        for model in self.available_models(task):
            try:
                result = self.train(
                    df,
                    target,
                    task,
                    model,
                )

                results.append({
                    "model": model,
                    **result["metrics"],
                })

            except Exception as exc:
                results.append({
                    "model": model,
                    "error": str(exc),
                })

        return results

