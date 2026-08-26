from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyzer import DatasetAnalyzer
from ml_engine import MLEngine
from visualizer import Visualizer
from report_generator import ReportGenerator


console = Console()


def banner():
    console.print(
        Panel.fit(
            "[bold cyan]AI DATA DETECTIVE[/bold cyan]\n"
            "[white]Automated Dataset Intelligence System[/white]",
            border_style="cyan",
        )
    )


def load_dataset():
    console.print("\n[bold]1.[/] Built-in sample")
    console.print("[bold]2.[/] Custom CSV")

    choice = input("Choose: ").strip()

    if choice == "1":
        path = Path("datasets/sample.csv")
    elif choice == "2":
        path = Path(
            input("Enter CSV path: ")
            .strip()
            .strip('"')
        )
    else:
        console.print("[red]Invalid choice.[/red]")
        return None

    if not path.exists():
        console.print("[red]File does not exist.[/red]")
        return None

    try:
        df = pd.read_csv(path)
        console.print(
            f"[green]Loaded {len(df):,} rows "
            f"and {len(df.columns)} columns.[/green]"
        )
        return df

    except Exception as exc:
        console.print(
            f"[red]Could not read CSV: {exc}[/red]"
        )
        return None


def show_profile(df):
    analyzer = DatasetAnalyzer(df)
    profile = analyzer.profile()

    table = Table(
        title="Dataset Profile"
    )

    table.add_column("Metric")
    table.add_column("Value")

    values = [
        ("Rows", profile["rows"]),
        ("Columns", profile["columns"]),
        ("Numerical", len(profile["numeric"])),
        ("Categorical", len(profile["categorical"])),
        ("Missing Cells", profile["missing"]),
        ("Duplicates", profile["duplicates"]),
    ]

    for metric, value in values:
        table.add_row(
            metric,
            str(value)
        )

    console.print(table)


def show_quality(df):
    analyzer = DatasetAnalyzer(df)
    report = analyzer.quality_report()

    console.print(
        Panel(
            "[bold cyan]DATA QUALITY ANALYSIS[/bold cyan]"
        )
    )

    if report["missing_columns"]:
        for column, count in report[
            "missing_columns"
        ].items():
            console.print(
                f"[yellow]Missing:[/] "
                f"{column} ? {count}"
            )
    else:
        console.print(
            "[green]No missing values.[/green]"
        )

    if report["constant_columns"]:
        console.print(
            "[yellow]Constant columns:[/yellow] "
            + ", ".join(
                report["constant_columns"]
            )
        )
    else:
        console.print(
            "[green]No constant columns.[/green]"
        )

    console.print("\n[bold]Outliers[/bold]")

    for column, data in report[
        "outliers"
    ].items():

        if data["count"]:
            console.print(
                f"{column}: "
                f"{data['count']} "
                f"({data['percent']:.2f}%)"
            )


def show_statistics(df):
    analyzer = DatasetAnalyzer(df)

    numeric = list(
        df.select_dtypes("number").columns
    )

    if not numeric:
        console.print(
            "[yellow]No numeric columns.[/yellow]"
        )
        return

    for index, column in enumerate(
        numeric,
        1
    ):
        console.print(
            f"{index}. {column}"
        )

    try:
        choice = int(
            input("Select column: ")
        )

        column = numeric[choice - 1]

    except (ValueError, IndexError):
        console.print(
            "[red]Invalid selection.[/red]"
        )
        return

    values = analyzer.numeric_stats(
        column
    )

    table = Table(
        title=f"Statistics — {column}"
    )

    table.add_column("Metric")
    table.add_column("Value")

    for key, value in values.items():
        table.add_row(
            key,
            f"{value:.4f}"
        )

    console.print(table)


def show_correlations(df):
    analyzer = DatasetAnalyzer(df)

    pairs = analyzer.strongest_correlations()

    if not pairs:
        console.print(
            "[yellow]Insufficient numeric columns.[/yellow]"
        )
        return

    table = Table(
        title="Strongest Correlations"
    )

    table.add_column("Feature A")
    table.add_column("Feature B")
    table.add_column("Correlation")

    for pair in pairs:
        table.add_row(
            pair["feature_a"],
            pair["feature_b"],
            f"{pair['correlation']:.3f}",
        )

    console.print(table)


def run_ml(df):
    engine = MLEngine()

    console.print(
        "\n[bold]1.[/] Classification"
    )
    console.print(
        "[bold]2.[/] Regression"
    )

    choice = input("Task: ").strip()

    if choice == "1":
        task = "classification"
    elif choice == "2":
        task = "regression"
    else:
        return

    console.print("\nColumns:")

    for column in df.columns:
        console.print(f" - {column}")

    target = input(
        "\nTarget column: "
    ).strip()

    if target not in df.columns:
        console.print(
            "[red]Target does not exist.[/red]"
        )
        return

    models = engine.available_models(
        task
    )

    console.print("\nModels:")

    for i, model in enumerate(
        models,
        1
    ):
        console.print(
            f"{i}. {model}"
        )

    try:
        choice = int(
            input("Model: ")
        )

        model_name = models[
            choice - 1
        ]

    except (ValueError, IndexError):
        console.print(
            "[red]Invalid model.[/red]"
        )
        return

    try:
        result = engine.train(
            df,
            target,
            task,
            model_name,
        )

        table = Table(
            title=f"{model_name} Results"
        )

        table.add_column("Metric")
        table.add_column("Value")

        for key, value in result[
            "metrics"
        ].items():
            table.add_row(
                key.upper(),
                f"{value:.4f}"
            )

        console.print(table)

        return result

    except Exception as exc:
        console.print(
            f"[red]Training failed: {exc}[/red]"
        )
        return None


def compare_models(df):
    engine = MLEngine()

    target = input(
        "Target column: "
    ).strip()

    if target not in df.columns:
        console.print(
            "[red]Target not found.[/red]"
        )
        return

    console.print(
        "1. Classification"
    )
    console.print(
        "2. Regression"
    )

    choice = input(
        "Task: "
    ).strip()

    task = (
        "classification"
        if choice == "1"
        else "regression"
    )

    results = engine.compare(
        df,
        target,
        task
    )

    table = Table(
        title="Model Comparison"
    )

    table.add_column("Model")

    metric_names = (
        ["accuracy", "precision", "recall", "f1"]
        if task == "classification"
        else ["mae", "rmse", "r2"]
    )

    for metric in metric_names:
        table.add_column(
            metric.upper()
        )

    for row in results:

        values = [row["model"]]

        for metric in metric_names:
            if metric in row:
                values.append(
                    f"{row[metric]:.4f}"
                )
            else:
                values.append("-")

        table.add_row(*values)

    console.print(table)


def generate_visuals(df):
    visualizer = Visualizer()

    paths = visualizer.generate_all(df)

    console.print(
        f"[green]Generated {len(paths)} plots.[/green]"
    )

    for path in paths:
        console.print(
            f"  {path}"
        )


def generate_report(df, model_result=None):
    analyzer = DatasetAnalyzer(df)

    quality = analyzer.quality_report()

    report = ReportGenerator()

    result = None

    if model_result:
        result = {
            "model": model_result["model"],
            "task": model_result["task"],
            "target": model_result["target"],
            "metrics": model_result["metrics"],
        }

    path = report.generate(
        df,
        quality,
        result
    )

    console.print(
        f"[green]Report saved:[/green] {path}"
    )


def application():
    dataframe = None
    last_model = None

    while True:

        console.print(
            "\n[bold cyan]"
            "1. Load Dataset\n"
            "2. Dataset Profile\n"
            "3. Data Quality\n"
            "4. Statistics\n"
            "5. Correlation Analysis\n"
            "6. Train ML Model\n"
            "7. Compare Models\n"
            "8. Generate Visualizations\n"
            "9. Generate Report\n"
            "0. Exit"
            "[/bold cyan]"
        )

        choice = input(
            "\nChoose: "
        ).strip()

        if choice == "1":
            dataframe = load_dataset()

        elif choice in {
            "2", "3", "4", "5",
            "6", "7", "8", "9"
        } and dataframe is None:
            console.print(
                "[yellow]Load a dataset first.[/yellow]"
            )

        elif choice == "2":
            show_profile(dataframe)

        elif choice == "3":
            show_quality(dataframe)

        elif choice == "4":
            show_statistics(dataframe)

        elif choice == "5":
            show_correlations(dataframe)

        elif choice == "6":
            last_model = run_ml(
                dataframe
            )

        elif choice == "7":
            compare_models(
                dataframe
            )

        elif choice == "8":
            generate_visuals(
                dataframe
            )

        elif choice == "9":
            generate_report(
                dataframe,
                last_model
            )

        elif choice == "0":
            console.print(
                "[cyan]AI Data Detective closed.[/cyan]"
            )
            break

        else:
            console.print(
                "[red]Invalid option.[/red]"
            )


if __name__ == "__main__":
    banner()
    application()

