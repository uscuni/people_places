import numpy as np
import pandas as pd

# specify paths to the different csv files
paths = [
    "Zensus_Bevoelkerung_100m-Gitter.csv",
    "Bevoelkerung100M.csv",
    "Haushalte100m.csv",
    "Familie100m.csv",
    "Geb100m.csv",
    "Wohnungen100m.csv",
]

# read translation excel
translations = pd.read_excel(
    "/home/lisa/work/Data_Format_Census.xlsx", sheet_name="Translations"
)
# tansform to dictionary with original as key and translation as value
translation_dict = translations.set_index("Original")["Translated"].to_dict()

for i in paths:
    # read in data as dataframes
    if i in ["Zensus_Bevoelkerung_100m-Gitter.csv", "Bevoelkerung100M.csv"]:
        data = pd.read_csv("/data/" + i, delimiter=";", encoding="cp1252")
    else:
        data = pd.read_csv("/data/" + i, delimiter=",", encoding="cp1252")

    if i == "Zensus_Bevoelkerung_100m-Gitter.csv":
        # replace all -1 with NaN
        data = data.replace(-1, np.nan)

        # rename columns
        data = data.rename(columns=translation_dict)

    else:
        # pivot dataframe to turn rows into columns
        data = data.pivot(
            index="Gitter_ID_100m",
            columns=["Merkmal", "Auspraegung_Text"],
            values="Anzahl",
        )

        # rename columns
        data = data.rename(columns=translation_dict)
        data = data.rename_axis(columns=translation_dict)
        data.index.names = ["ID"]

        coords = data.index.str.extract(r"N(\d{5})E(\d{5})")
        coords[0] = pd.to_numeric(coords[0]) * 100 + 50
        coords[1] = pd.to_numeric(coords[1]) * 100 + 50

        data[[("", "N"), ("", "E")]] = coords.values

    data.to_parquet("/data/processed_data/" + i[:-4] + ".parquet")
