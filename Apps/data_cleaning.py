# FIXME: Needs some improvement
from data_loading import LoadedData


class CleanedData:
    """Class for cleaning and storing the cleaned data"""

    # Removal of the irrelevant data
    for iter1 in LoadedData.test_data.columns.to_list():
        if LoadedData.train_data[iter1].isin(LoadedData.test_data[iter1].unique()).sum() == 0\
                and iter1 != 'protein_sequence':
            if 'cleaned_train_data' not in locals():
                cleaned_train_data = LoadedData.train_data.drop(iter1, axis=1, )
                cleaned_test_data = LoadedData.test_data.drop(iter1, axis=1)

            else:
                cleaned_train_data.drop(iter1, axis=1, inplace=True)
                cleaned_test_data.drop(iter1, axis=1, inplace=True)

    # Removal of the empty tm train data rows
    cleaned_train_data.drop(cleaned_train_data[cleaned_train_data['tm'].isna()].index, inplace=True)

    # Removal of the empty pH train data rows (letter can be replaced with the average)
    cleaned_train_data.drop(cleaned_train_data[cleaned_train_data['pH'].isna()].index, inplace=True)


if __name__ == '__main__':
    pass
