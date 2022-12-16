from Apps.data_loading import LoadedData
from pandas import DataFrame


class CleanedData(LoadedData):
    """Class for cleaning and storing the cleaned data"""

    @staticmethod
    def _remove_empty_rows(df: DataFrame, col_name: str) -> None:
        df.drop(df[df[col_name].isna()].index, inplace=True)


    def __init__(self, target_column: str = 'tm') -> None:
        super(CleanedData, self).__init__()
        # Removal of the irrelevant data
        for col_name in self.test_df.columns.to_list():
            if self.train_df[col_name].isin(self.test_df[col_name].unique()).sum() == 0\
                    and col_name != 'protein_sequence':
                if 'cleaned_train_data' not in locals():
                    self.cleaned_train_data = self.train_df.drop(col_name, axis=1, )
                    self.cleaned_test_data = self.test_df.drop(col_name, axis=1)

                else:
                    self.cleaned_train_data.drop(col_name, axis=1, inplace=True)
                    self.cleaned_test_data.drop(col_name, axis=1, inplace=True)

        self._remove_empty_rows(self.cleaned_train_data, target_column)

        del self.test_df, self.train_df


if __name__ == '__main__':
    Cleaned_data = CleanedData()
    print(Cleaned_data.cleaned_train_data.head())
    print(Cleaned_data.cleaned_train_data.shape)
    print(Cleaned_data.cleaned_test_data.head())
    print(Cleaned_data.cleaned_test_data.shape)
    pass
