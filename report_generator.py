from datetime import datetime
from pathlib import Path


class ReportGenerator:
    def __init__(self, output="reports"):
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)

    def generate(
        self,
        df,
        quality_report,
        model_result=None
    ):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        path = self.output / (
            f"data_detective_{timestamp}.md"
        )

        profile = quality_report["profile"]

        lines = [
            "# AI Data Detective Report\n\n",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n",
            "## Dataset Overview\n\n",
            f"- Rows: {profile['rows']}\n",
            f"- Columns: {profile['columns']}\n",
            f"- Numerical columns: "
            f"{len(profile['numeric'])}\n",
            f"- Categorical columns: "
            f"{len(profile['categorical'])}\n",
            f"- Missing cells: "
            f"{profile['missing']}\n",
            f"- Duplicate rows: "
            f"{profile['duplicates']}\n",
            "\n## Missing Values\n\n",
        ]

        if quality_report["missing_columns"]:
            for col, count in quality_report[
                "missing_columns"
            ].items():
                lines.append(
                    f"- {col}: {count}\n"
                )
        else:
            lines.append("- None detected.\n")

        lines.append(
            "\n## Constant Columns\n\n"
        )

        constants = quality_report[
            "constant_columns"
        ]

        if constants:
            for col in constants:
                lines.append(f"- {col}\n")
        else:
            lines.append("- None detected.\n")

        lines.append(
            "\n## Outlier Summary\n\n"
        )

        for col, info in quality_report[
            "outliers"
        ].items():
            lines.append(
                f"- {col}: "
                f"{info['count']} "
                f"({info['percent']:.2f}%)\n"
            )

        if model_result:
            lines.extend([
                "\n## Machine Learning\n\n",
                f"- Task: {model_result['task']}\n",
                f"- Model: {model_result['model']}\n",
                f"- Target: {model_result['target']}\n",
                "\n### Metrics\n\n",
            ])

            for key, value in model_result[
                "metrics"
            ].items():
                lines.append(
                    f"- {key.upper()}: {value:.4f}\n"
                )

        path.write_text(
            "".join(lines),
            encoding="utf-8"
        )

        return path

