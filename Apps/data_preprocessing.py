# FIXME: Conversion into feature extraction class
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from feature_engineering import FeatureData


class PreprocessedData(FeatureData):

    cat_cols = [col_name for col_name in FeatureData.train_X.columns if
                FeatureData.train_X[col_name].nunique() < 10 and
                FeatureData.train_X[col_name].dtype == "object"]

    num_cols = [col_name for col_name in FeatureData.train_X.columns if
                FeatureData.train_X[col_name].dtype in ['int64', 'float64'] and
                (FeatureData.train_X[col_name].min() < 0 or FeatureData.train_X[col_name].max() > 1)]


    # Preprocessing for numerical data
    if len(num_cols) >= 1:
        num_trans = MinMaxScaler(copy=False)

    # Preprocessing for categorical data
    if len(cat_cols) >= 1:
        cat_trans = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    # Bundle preprocessing for numerical and categorical data
    if all(['cat_trans', 'num_trans']) in locals():
        preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_cols), ('cat', cat_trans, cat_cols)])

    elif 'cat_trans' in locals():
        preprocessor = ColumnTransformer(transformers=[('cat', cat_trans, cat_cols)])

    elif 'num_trans' in locals():
        preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_cols)])

    else:
        preprocessor = None


if __name__ == '__main__':
    print(PreprocessedData.preprocessor)
    pass
