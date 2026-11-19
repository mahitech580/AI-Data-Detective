import pandas as pd

from analyzer import DatasetAnalyzer


def test_profile_recognizes_string_extension_and_category_columns():
    frame = pd.DataFrame({
        "segment": pd.Series(["north", "south", "north"], dtype="string"),
        "tier": pd.Series(["basic", "plus", "basic"], dtype="category"),
        "active": pd.Series([True, False, True], dtype="boolean"),
        "amount": [10.0, 20.0, 30.0],
    })

    profile = DatasetAnalyzer(frame).profile()

    assert set(profile["categorical"]) == {"segment", "tier", "active"}
    assert profile["numeric"] == ["amount"]
    assert profile["rows"] == 3
    assert profile["columns"] == 4
