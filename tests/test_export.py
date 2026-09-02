from io import BytesIO

import pandas as pd

from trialsleuth.export import dataframe_to_safe_csv


def test_csv_export_neutralises_spreadsheet_formula_prefixes() -> None:
    dataframe = pd.DataFrame(
        {
            "Column(s)": ["=CMD()", "+SUM(1,1)", "-1+2", "@malicious", "safe_name"],
            "Affected rows": [1, 2, 3, 4, 5],
        }
    )

    exported = pd.read_csv(BytesIO(dataframe_to_safe_csv(dataframe)))

    assert exported["Column(s)"].tolist() == [
        "'=CMD()",
        "'+SUM(1,1)",
        "'-1+2",
        "'@malicious",
        "safe_name",
    ]
