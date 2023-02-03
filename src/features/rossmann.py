import pandas as pd


def add_rossmann_features(data: pd.DataFrame) -> pd.DataFrame:
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), "Open"] = 1
    # Use some properties directly
    features = [
        "Store",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]

    # add some more with a bit of preprocessing
    features.append("SchoolHoliday")
    data["SchoolHoliday"] = data["SchoolHoliday"].astype(float)

    # features.append('StateHoliday')
    # data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    # data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    # data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    # data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append("DayOfWeek")
    features.append("month")
    features.append("day")
    features.append("year")

    data["year"] = data.Date.apply(lambda x: x.split("-")[0])
    data["year"] = data["year"].astype(int)
    data["month"] = data.Date.apply(lambda x: x.split("-")[1])
    data["month"] = data["month"].astype(int)
    data["day"] = data.Date.apply(lambda x: x.split("-")[2])
    data["day"] = data["day"].astype(int)

    features.append("StoreType")
    data.loc[data["StoreType"] == "a", "StoreType"] = "1"
    data.loc[data["StoreType"] == "b", "StoreType"] = "2"
    data.loc[data["StoreType"] == "c", "StoreType"] = "3"
    data.loc[data["StoreType"] == "d", "StoreType"] = "4"
    data["StoreType"] = data["StoreType"].astype(int)

    features.append("Assortment")
    data.loc[data["Assortment"] == "a", "Assortment"] = "1"
    data.loc[data["Assortment"] == "b", "Assortment"] = "2"
    data.loc[data["Assortment"] == "c", "Assortment"] = "3"
    data["Assortment"] = data["Assortment"].astype(int)

    data = data[features + ["Sales"]]
    return data
