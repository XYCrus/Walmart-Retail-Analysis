import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def plot_combined_numeric(data, numeric_features):
    plt.figure(figsize=(12, 8))
    data[numeric_features].plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False)
    plt.tight_layout()
    plt.suptitle('Box Plots of Numeric Features', fontsize=16, y=1.02)
    plt.show()

def plot_unique_counts_categorical(data, categorical_features):
    unique_counts = {feature: data[feature].nunique() for feature in categorical_features}
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(unique_counts.keys()), y=list(unique_counts.values()))
    plt.yscale('log')  
    plt.title('Number of Unique Values per Categorical Feature (Log Scale)')
    plt.ylabel('Unique Values Count (Log Scale)')
    plt.xlabel('Categorical Features')
    plt.xticks(rotation=45)
    plt.show()

def main():
    data = pd.read_csv('../data/WalmartCombined.csv')

    data.replace('\\N', np.nan, inplace=True)
    data = data.dropna()

    numeric_features = ['customer_age', 'discount', 'order_date_year', 'product_base_margin', 'sales', 'unit_price', 'zip_code']
    for feature in numeric_features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

    categorical_features = ['city', 'customer_segment', 'product_category', 'product_container', 'region', 'state']

    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col, encoder in label_encoders.items():
        data[col] = encoder.fit_transform(data[col])

    data = data.dropna()

    plot_combined_numeric(data, numeric_features)
    plot_unique_counts_categorical(data, categorical_features)

if __name__ == "__main__":
    main()