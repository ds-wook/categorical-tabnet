# %%
import pandas as pd

# %%
df = pd.read_csv("../input/credit-card-customers/BankChurners.csv")
df = df.drop(
    [
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    ],
    axis=1,
)
df.head()
# %%
income_category_map = {
    "Less than $40K": 0,
    "$40K - $60K": 1,
    "$60K - $80K": 2,
    "$80K - $120K": 3,
    "$120K +": 4,
    "Unknown": 5,
}


card_category_map = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}


attrition_flag_map = {"Existing Customer": 0, "Attrited Customer": 1}

education_level_map = {
    "Uneducated": 0,
    "High School": 1,
    "College": 2,
    "Graduate": 3,
    "Post-Graduate": 4,
    "Doctorate": 5,
    "Unknown": 6,
}


df["Income_Category"] = df["Income_Category"].map(income_category_map)
df["Card_Category"] = df["Card_Category"].map(card_category_map)
df["Attrition_Flag"] = df["Attrition_Flag"].map(attrition_flag_map)
df["Education_Level"] = df["Education_Level"].map(education_level_map)

# %%
df.head()
# %%
cat_cols = [x for x in df.columns if df[x].dtype == 'object']
print(cat_cols)
# %%
