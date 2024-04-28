#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

#%%
def main():
    df1923 = pd.read_csv('../data/Walmart1923.csv',
                        on_bad_lines = 'skip')

    df1215 = pd.read_excel('../data/Walmart1215.csv')

    def standardize_column_names(df):
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    df1923 = standardize_column_names(df1923)
    df1215 = standardize_column_names(df1215)

    '''print(df1923.columns)
    print(df1215.columns)'''

    common_columns = df1923.columns.intersection(df1215.columns)

    df1923_common = df1923[common_columns]
    df1215_common = df1215[common_columns]

    combined_df = pd.concat([df1923_common, df1215_common], ignore_index=True)

    combined_df.to_csv('../data/WalmartCombined.csv', index=False)

    ###################
    data = pd.read_csv('../data/WalmartCombined.csv')

    data.replace('\\N', np.nan, inplace=True)

    data['order_date'] = pd.to_datetime(data['order_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data['ship_date'] = pd.to_datetime(data['ship_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    data = data.drop(columns=['customer_name', 'order_id'])

    numeric_columns = ['customer_age', 'discount', 'order_quantity', 'product_base_margin', 'sales', 'shipping_cost', 'unit_price']

    data = data.dropna()

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data['profit'] = pd.to_numeric(data['profit'], errors='coerce')

    X = data.drop('profit', axis=1)
    y = data['profit']

    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    numeric_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col != 'profit']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=0))])

    X_train, X_test, y_train, y_test = train_test_split(data.drop('profit', axis=1), data['profit'], test_size=0.3, random_state=777)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, predictions))
    print(f'Model RMSE: {rmse}')
    print("Model R^2 Score:", model.score(X_test, y_test))
    
    feature_names = [c.replace(' ', '_') for c in X.columns]

    explainer = LimeTabularExplainer(X_train.values,
                                     feature_names=feature_names,
                                     class_names=['profit'],
                                     categorical_features=categorical_cols,
                                     verbose=True,
                                     mode='regression')

    instance_index = 0 
    instance = X_test.iloc[instance_index].values

    exp = explainer.explain_instance(instance, model.predict, num_features=5)

    exp.show_in_notebook(show_all=False)

    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.show()



if __name__ == "__main__":
    main()
