from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numpy import arange
from pandas import DataFrame, Series
from data_loading import LoadedData
from matplotlib.pyplot import figure, barh, title, yticks, show
from data_preprocessing import PreprocessedData


class ModelTraining(PreprocessedData):

    def _make_mi_scores(self):
        mi_scores = mutual_info_regression(self.X_train, self.y_train)
        mi_scores = Series(mi_scores, name="MI Scores", index=self.data_dict['col_labels'])
        # mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def _plot_mi_scores(self, scores):
        # scores = scores.sort_values(ascending=True)
        width = arange(len(scores))
        ticks = list(scores.index)
        figure(dpi=100, figsize=(20, 15))
        barh(width, scores)
        yticks(width, ticks)
        title("Mutual Information Scores")
        show()

    def _plot_mi_scores2(self, scores):
        scores.index = scores.index.map(lambda x: x.split("_")[0])
        scores = scores.groupby(level=0).mean()
        width = arange(len(scores))
        ticks = list(scores.index)
        figure(dpi=100, figsize=(20, 15))
        barh(width, scores)
        yticks(width, ticks)
        title("Mutual Information Scores")
        show()

    def get_PCA(self):
        scaling = StandardScaler()
        scaling.fit(self.X_train)
        self.X_train = scaling.transform(self.X_train)
        self.X_test = scaling.transform(self.X_test)

        pca = PCA()
        X_pca = pca.fit_transform(self.X_train)

        component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
        X_pca = DataFrame(X_pca, columns=component_names)
        print(X_pca.head())

        loadings = DataFrame(
            pca.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=self.data_dict['col_labels'],  # and the rows are the original features
        )
        print(loadings)

        mi_scores = mutual_info_regression(X_pca, self.y_train, discrete_features=False)
        mi_scores = Series(mi_scores, name="MI Scores", index=component_names)
        mi_scores = mi_scores.sort_values(ascending=False)

        print(mi_scores)


    def __init__(self):
        super(ModelTraining, self).__init__()


        # self.model = XGBRegressor(subsample=0.6,
        #                           n_estimators=500,
        #                           max_depth=6,
        #                           learning_rate=0.01,
        #                           colsample_bytree=0.7999999999999999,
        #                           colsample_bylevel=0.5,
        #                           early_stopping_rounds=20,
        #                           seed=20)

        self.model = XGBRegressor(subsample=0.6,
                                  n_estimators=1500,
                                  max_depth=10,
                                  learning_rate=0.03,
                                  colsample_bytree=0.7999999999999999,
                                  colsample_bylevel=0.5,
                                  early_stopping_rounds=10,
                                  seed=20)



        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_dict['train_X'],
                                                                                self.data_dict['train_y'],
                                                                                test_size=0.2,
                                                                                random_state=123)


        self.model.fit(self.X_train, self.y_train,
                       eval_set=[(self.X_test, self.y_test)],
                       verbose=False)

        self.y_pred = self.model.predict(self.data_dict['test_X'])
        self.y_pred2 = self.model.predict(self.X_test)

        self.score = mean_absolute_error(self.y_test, self.y_pred2)
        self.score2 = r2_score(self.y_test, self.y_pred2)


if __name__ == '__main__':
    model_training = ModelTraining()
    print(model_training._make_mi_scores())
    # print(model_training._plot_mi_scores(model_training._make_mi_scores()))
    print(model_training._plot_mi_scores2(model_training._make_mi_scores()))
    print(model_training.get_PCA())
    print(model_training._make_mi_scores())

    print(spearmanr(model_training.y_test, model_training.y_pred2))
    print(model_training.score)

    submission_data = DataFrame(model_training.data_dict['original_test_indices'], columns=['seq_id'])
    submission_data['tm'] = DataFrame(model_training.y_pred, columns=['tm'])

    submission_data.to_csv("submission.csv", index=False)

    print(model_training.score2)
    pass
