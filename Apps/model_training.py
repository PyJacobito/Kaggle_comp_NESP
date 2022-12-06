# FIXME: Training the model on the whole pool of the data
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numpy import arange

from data_preprocessing import PreprocessedData
from feature_engineering import FeatureData


class ModelTraining:

    # Model definition
    # model = XGBRegressor(scale_pos_weight=1,
    #                      learning_rate=0.01,
    #                      colsample_bytree=0.4,
    #                      subsample=0.8,
    #                      objective='reg:squarederror',
    #                      eval_metric=mean_absolute_error,
    #                      n_estimators=1000,
    #                      reg_alpha=0.3,
    #                      max_depth=4,
    #                      gamma=10)

    model = XGBRegressor(subsample=0.6,
                         n_estimators=500,
                         max_depth=6,
                         learning_rate=0.01,
                         colsample_bytree=0.7999999999999999,
                         colsample_bylevel=0.5,
                         seed=20)


    # Model pipeline
    if PreprocessedData.preprocessor is not None:
        model_pipeline = Pipeline(steps=[('preprocessor', PreprocessedData.preprocessor),
                                         ('model', model)
                                         ])
    else:
        model_pipeline = model

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(FeatureData.train_X,
                                                        FeatureData.train_Y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Tuning
    # params = {'max_depth': [3, 6, 10],
    #           'learning_rate': [0.01, 0.05, 0.1],
    #           'n_estimators': [100, 500, 1000],
    #           'colsample_bytree': [0.3, 0.7]}

    # params = {'max_depth': [3, 5, 6, 10, 15, 20],
    #           'learning_rate': [0.01, 0.1, 0.2, 0.3],
    #           'subsample': arange(0.5, 1.0, 0.1),
    #           'colsample_bytree': arange(0.4, 1.0, 0.1),
    #           'colsample_bylevel': arange(0.4, 1.0, 0.1),
    #           'n_estimators': [100, 500, 1000]}

    # Definition of model selection
    # clf = RandomizedSearchCV(estimator=model,
    #                          param_distributions=params,
    #                          scoring='neg_mean_squared_error',
    #                          n_iter=25,
    #                          verbose=1)

    # # Preprocessing of training data, fit model
    model_pipeline.fit(X_train, y_train)

    # Fiting of the model selection
    # clf.fit(X_train, y_train)
    # print("Best parameters:", clf.best_params_)
    # print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))


    # # Preprocessing of validation data, get predictions
    y_pred = model_pipeline.predict(X_test)

    # # Evaluate the model
    score = mean_absolute_error(y_test, y_pred)


if __name__ == '__main__':
    print(ModelTraining.score)
    print('\n')
    print(spearmanr(ModelTraining.y_test, ModelTraining.y_pred))
    pass
