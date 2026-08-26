import numpy as np
import pandas as pd
from scipy import stats


class DatasetAnalyzer:
    def __init__(self, df):
        self.df = df

    def profile(self):
        numeric = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric": numeric,
            "categorical": categorical,
            "missing": int(self.df.isna().sum().sum()),
            "duplicates": int(self.df.duplicated().sum()),
        }

    def column_profile(self):
        rows = []

        for col in self.df.columns:
            s = self.df[col]

            rows.append({
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 2),
                "unique": int(s.nunique(dropna=True)),
            })

        return pd.DataFrame(rows)

    def numeric_stats(self, column):
        if column not in self.df:
            raise ValueError("Column not found.")

        s = pd.to_numeric(
            self.df[column],
            errors="coerce"
        ).dropna()

        if s.empty:
            raise ValueError("Column contains no numeric values.")

        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "variance": float(s.var()),
            "min": float(s.min()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skewness": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
        }

    def correlations(self):
        return self.df.select_dtypes(
            include=np.number
        ).corr()

    def strongest_correlations(self, limit=10):
        matrix = self.correlations()
        pairs = []
        columns = list(matrix.columns)

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                value = matrix.iloc[i, j]

                if pd.notna(value):
                    pairs.append({
                        "feature_a": columns[i],
                        "feature_b": columns[j],
                        "correlation": float(value),
                    })

        return sorted(
            pairs,
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )[:limit]

    def outliers(self):
        result = {}

        for col in self.df.select_dtypes(include=np.number).columns:
            s = self.df[col].dropna()

            if len(s) < 4:
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(((s < lower) | (s > upper)).sum())

            result[col] = {
                "count": count,
                "percent": round(count / len(s) * 100, 2),
            }

        return result

    def quality_report(self):
        return {
            "profile": self.profile(),
            "missing_columns": {
                col: int(count)
                for col, count in self.df.isna().sum().items()
                if count > 0
            },
            "constant_columns": [
                col
                for col in self.df.columns
                if self.df[col].nunique(dropna=False) <= 1
            ],
            "outliers": self.outliers(),
        }

