---
layout: post
title: "Data Wrangling the Titanic Dataset"
date: 2026-09-28 09:00:00 +1000
permalink: /blog/titanic-data-wrangling/
tags: [python, pandas, data-science]
description: "A practice walkthrough of cleaning and feature-engineering the classic Titanic dataset with pandas"
---

In this post, I am going through my practice in data wrangling on the classic Titanic dataset.

## Loading the data

First, we import the required libraries and load the dataset. We use a URL so this code works immediately, even if you haven't downloaded the CSV.

```python
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 30)

MIRROR_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
raw = pd.read_csv(MIRROR_URL)
df = raw.copy()
```

Before changing anything, we check the column data types and the count of non-null values to see exactly where our missing data is hiding.

```python
df.info()
```

```
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    str    
 4   Sex          891 non-null    str    
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    str    
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    str    
 11  Embarked     889 non-null    str    
dtypes: float64(2), int64(5), str(5)
memory usage: 83.7 KB
```

We also calculate summary statistics for our numeric columns. This helps us spot outliers and illogical values—such as a minimum `Fare` of 0, which is highly suspicious for a commercial ocean liner.

```python
df.describe().round(2)
```

|       | PassengerId | Survived | Pclass | Age    | SibSp | Parch | Fare   |
|-------|-------------|----------|--------|--------|-------|-------|--------|
| count | 891.00      | 891.00   | 891.00 | 714.00 | 891.00| 891.00| 891.00 |
| mean  | 446.00      | 0.38     | 2.31   | 29.70  | 0.52  | 0.38  | 32.20  |
| std   | 257.35      | 0.49     | 0.84   | 14.53  | 1.10  | 0.81  | 49.69  |
| min   | 1.00        | 0.00     | 1.00   | 0.42   | 0.00  | 0.00  | 0.00   |
| 25%   | 223.50      | 0.00     | 2.00   | 20.12  | 0.00  | 0.00  | 7.91   |
| 50%   | 446.00      | 0.00     | 3.00   | 28.00  | 0.00  | 0.00  | 14.45  |
| 75%   | 668.50      | 1.00     | 3.00   | 38.00  | 1.00  | 0.00  | 31.00  |
| max   | 891.00      | 1.00     | 3.00   | 80.00  | 8.00  | 6.00  | 512.33 |

## Utilising the name column

The `Name` column looks like standard text, but each name actually contains a title (like *Mr*, *Mrs*, *Miss*, or *Master*). These titles encode sex, rough age (a "Master" was a young boy), and social status. Historically, extracting these titles is crucial because it directly captures the "women and children first" policy—titles like Mrs, Miss, and Master survived at significantly higher rates than Mr.

We use a regular expression to extract the title. Because rare categories can confuse machine learning models later, we map some titles together and combine uncommon ones (like *Dr* or *Countess*) into a single `Rare` category.

```python
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
common_titles = {"Mr", "Mrs", "Miss", "Master"}

df["Title"] = df["Title"].replace(title_map)
df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")
```

## Handling the missing values

Only two passengers are missing an `Embarked` port. For such a small gap, it is safe to fill them with the dataset's most common port (the mode).

```python
most_common_port = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(most_common_port)
```

Missing ages require more care. Filling every missing age with the overall median (28) would falsely assign an adult age to a young boy. Instead, we generate a boolean flag to remember which ages were imputed, and then fill the gaps using the median age of passengers who share the exact same title and ticket class.

```python
df["AgeImputed"] = df["Age"].isna()

group_median_age = df.groupby(["Title", "Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(group_median_age)
```

The `Cabin` column is missing over 77% of its data, making it impossible to impute fully. However, the presence of a cabin and the deck letter (the first character of the cabin code) are valuable.

We extract these into new features.

Also, one passenger is listed on deck "T", which didn't actually exist on the ship's passenger deck plans. Assuming it is a data entry error, we combine it into our "Unknown" category.

```python
df["HasCabin"] = df["Cabin"].notna()
df["Deck"] = df["Cabin"].str[0].fillna("Unknown").replace({"T": "Unknown"})
```

Earlier, we noticed fares of exactly 0. Historically, several of these passengers shared `LINE` tickets and were actually ship crew or company employees traveling for free. A £0 first-class fare isn't comparable to what a paying passenger spent. We flag these free tickets, treat their fare as missing, and impute it using the median fare for their class.

```python
df["FreeTicket"] = df["Fare"] == 0
df["Fare"] = df["Fare"].replace(0, np.nan)
df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
```

Furthermore, Titanic tickets were often purchased for entire families rather than individuals, meaning the `Fare` column frequently represents a group total. To fix this skew, we count how many passengers share the same ticket number and divide the fare by that group size to calculate the true price per person.

```python
df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
df["FarePerPerson"] = (df["Fare"] / df["TicketGroupSize"]).round(2)
```

## Some data modification

We can combine `SibSp` (siblings/spouses) and `Parch` (parents/children), plus the passenger themselves, to calculate their total family size on board. We then use `pd.cut()` to group these into intuitive categories: Solo travelers, Small families, and Large families.

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = df["FamilySize"] == 1

df["FamilyType"] = pd.cut(
    df["FamilySize"],
    bins=[0, 1, 4, 11],
    labels=["Alone", "Small", "Large"]
)
```

Similarly, we group the continuous `Age` variable into discrete life stages. This makes it much easier to plot survival rates for "Teens" versus "Seniors" later on.

```python
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young adult", "Adult", "Senior"]
)
```

## Comparing the original and cleaned data

```python
print("--- ORIGINAL DATA ---")
display(raw.head(3))

print("\n--- CLEANED DATA ---")
display(df.head(3))
```

**Original data:**

| PassengerId | Survived | Pclass | Name                                             | Sex    | Age  | SibSp | Parch | Ticket           | Fare    | Cabin | Embarked |
|-------------|----------|--------|--------------------------------------------------|--------|------|-------|-------|------------------|---------|-------|----------|
| 1           | 0        | 3      | Braund, Mr. Owen Harris                          | male   | 22.0 | 1     | 0     | A/5 21171        | 7.2500  | NaN   | S        |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Th...| female | 38.0 | 1     | 0     | PC 17599         | 71.2833 | C85   | C        |
| 3           | 1        | 3      | Heikkinen, Miss. Laina                            | female | 26.0 | 0     | 0     | STON/O2. 3101282 | 7.9250  | NaN   | S        |

**Cleaned data:**

| PassengerId | Survived | Pclass | Sex    | Age  | Title | AgeImputed | HasCabin | Deck    | FreeTicket | TicketGroupSize | FarePerPerson | FamilySize | IsAlone | FamilyType | AgeGroup    |
|-------------|----------|--------|--------|------|-------|------------|----------|---------|------------|------------------|----------------|------------|---------|------------|-------------|
| 1           | 0        | 3      | male   | 22.0 | Mr    | False      | False    | Unknown | False      | 1                | 7.25           | 2          | False   | Small      | Young adult |
| 2           | 1        | 1      | female | 38.0 | Mrs   | False      | True     | C       | False      | 1                | 71.28          | 2          | False   | Small      | Adult       |
| 3           | 1        | 3      | female | 26.0 | Miss  | False      | False    | Unknown | False      | 1                | 7.92           | 1          | True    | Alone      | Young adult |

Starting from 12 raw columns, we now have a dataset with model-ready features, including titles that capture social status, imputed ages, per-person fares that remove the family ticket skew, and clean categorical groupings for family size and age.
