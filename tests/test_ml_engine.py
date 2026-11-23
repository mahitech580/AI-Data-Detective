import pandas as pd

from ml_engine import MLEngine


def test_classification_uses_unstratified_split_when_test_partition_is_too_small():
    frame = pd.DataFrame({
        "signal": range(30),
        "target": [f"class-{index // 3}" for index in range(30)],
    })

    result = MLEngine().train(
        frame,
        target="target",
        task="classification",
        model_name="Decision Tree",
    )

    assert len(result["predictions"]) == 6
    assert result["metrics"]["accuracy"] >= 0
