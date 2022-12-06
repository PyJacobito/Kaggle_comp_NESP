# FIXME: creation of the path function with the decorator handler
from os import path, sep, pardir, getcwd
from typing import AnyStr

from pandas import DataFrame, read_csv, concat


def _get_loading_path(path_str: AnyStr) -> AnyStr:
    """Function returns the path to data in the working directory"""
    return str(path.normpath(getcwd() + sep + pardir)) + "\\Docs\\Kaggle_data_sample\\" + path_str


def _get_corrected_train_data() -> DataFrame:
    """Function returns the corrected train data with the last update file"""
    train_df = read_csv(_get_loading_path("train.csv"), index_col="seq_id")
    update_df = read_csv(_get_loading_path("train_updates_20220929.csv"), index_col="seq_id")
    update_df.data_source = train_df.loc[update_df.index, 'data_source']
    update_indexes = update_df[update_df.protein_sequence.isna()].index
    update_df.loc[update_indexes, 'protein_sequence'] = train_df.loc[update_indexes, 'protein_sequence']
    train_df.drop(update_df.index, inplace=True)
    return concat([train_df, update_df])


def _get_test_data() -> DataFrame:
    """Function returns the test data"""
    return read_csv(_get_loading_path('test.csv'), index_col="seq_id")


class LoadedData:
    """Class storing the train and test data"""
    train_data = _get_corrected_train_data()
    test_data = _get_test_data()


if __name__ == "__main__":
    pass
