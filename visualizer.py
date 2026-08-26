from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, output="generated_plots"):
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)

    def generate_histograms(self, df):
        paths = []

        numeric = df.select_dtypes("number").columns

        for column in numeric:
            path = self.output / f"{column}_distribution.png"

            plt.figure(figsize=(8, 5))
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f"{column} Distribution")
            plt.tight_layout()
            plt.savefig(path, dpi=140)
            plt.close()

            paths.append(path)

        return paths

    def correlation_heatmap(self, df):
        numeric = df.select_dtypes("number")

        if numeric.shape[1] < 2:
            return None

        path = self.output / "correlation_heatmap.png"

        plt.figure(
            figsize=(
                max(7, numeric.shape[1]),
                max(5, numeric.shape[1]),
            )
        )

        sns.heatmap(
            numeric.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

        return path

    def generate_all(self, df):
        paths = self.generate_histograms(df)

        heatmap = self.correlation_heatmap(df)

        if heatmap:
            paths.append(heatmap)

        return paths

