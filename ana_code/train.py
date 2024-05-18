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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import shap
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score


#%%
def preprocess():
    df1923 = pd.read_csv('../data/Walmart1923.csv',
                        on_bad_lines = 'skip')

    df1215 = pd.read_excel('../data/Walmart1215.csv')

    def standardize_column_names(df):
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    df1923 = standardize_column_names(df1923)
    df1215 = standardize_column_names(df1215)

    common_columns = df1923.columns.intersection(df1215.columns)

    df1923_common = df1923[common_columns]
    df1215_common = df1215[common_columns]

    combined_df = pd.concat([df1923_common, df1215_common], ignore_index=True)

    combined_df.to_csv('../data/WalmartCombined.csv', index=False)

def main():
    
    data = pd.read_csv('../data/WalmartCombined.csv')

    data.replace('\\N', np.nan, inplace=True)
    data = data.dropna()

    data['customer_age'] = pd.to_numeric(data['customer_age'], errors='coerce')
    data['discount'] = pd.to_numeric(data['discount'], errors='coerce')
    data['order_date_year'] = pd.to_numeric(data['order_date_year'], errors='coerce')
    data['product_base_margin'] = pd.to_numeric(data['product_base_margin'], errors='coerce')
    data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
    data['unit_price'] = pd.to_numeric(data['unit_price'], errors='coerce')
    data['zip_code'] = pd.to_numeric(data['zip_code'], errors='coerce')


    categorical_features = ['city', 'customer_segment', 'product_category', 'product_container', 'region', 'state']
    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col, encoder in label_encoders.items():
        data[col] = encoder.fit_transform(data[col])

    data = data.dropna()

    X = data.drop(columns=['profit'])  
    y = data['profit']

    scaler = StandardScaler()
    y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('start')

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',  
        'max_depth': 6,                   
        'eta': 0.1,                     
        'colsample_bytree': 0.8,        
        'subsample': 0.8,                 
        'eval_metric': 'rmse',            
        'n_jobs': -1,                     
        'seed': 42 
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=10000, evals=evals)

    y_pred = model.predict(dtest)
    

    # Calculate RMSE and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse}")
    print(f"Test R-squared: {r2}")


    explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, mode='regression')

    # Choose an instance to explain
    instance = X_test.iloc[0].values
    lime_explanation = explainer_lime.explain_instance(instance, model.predict, num_features=10)

    # Visualize the LIME explanation
    lime_explanation.show_in_notebook()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)

    shap.dependence_plot(shap_values, X_test)

    

if __name__ == "__main__":
    main()
