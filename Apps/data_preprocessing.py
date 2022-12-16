from pandas import DataFrame, concat
from protlearn.features import aac, entropy, aaindex1, atc
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
# from lev_dist_func import levenshtein_ratio_and_distance
from typing import List, AnyStr, Any
from numpy import ndarray, array, transpose, concatenate

from data_cleaning import CleanedData


class FeatureData(CleanedData):

    __ami_symbols = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    __seq_col = 'protein_sequence'
    __target_column = 'tm'

    @staticmethod
    def _ftn1_convert_num_columns(df: DataFrame, col_names: List[AnyStr], *args: Any) -> ndarray:
        return df[col_names].to_numpy()

    @staticmethod
    def _ftc1_get_seq_length(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        return df[col_name].map(lambda x: len(x)).to_numpy()

    @staticmethod
    def _ftc2_get_count_symb(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col,
                             amino_res_list: List[AnyStr] = __ami_symbols) -> ndarray:
        return array(
            [[df[col_name].map(lambda x: x.count(amino_symbol)).tolist()] for amino_symbol in amino_res_list]).T


    @staticmethod
    def _join_arr(func_list: List[AnyStr], *args: Any) -> ndarray:
        pass
        # return concatenate(*fun)

    # @staticmethod
    # def _ftc3_get_

    def __init__(self):
        super(FeatureData, self).__init__()

        for df_purpose in ['train', 'test']:
            for df_type in ['Y', 'X']:
                df_name = f'{df_purpose}_{df_type}'
                if df_type == 'Y':
                    self.df_name = getattr(self, f'cleaned_{df_name}_data')[self.__target_column].to_numpy

                else:
                    self.df_name = getattr(self, f'cleaned_{df_name}_data').drop(self.__target_column, axis=1, inplace=True)

        self.cat_col_list = [col_name for col_name in self.cleaned_train_data.columns
                             if self.cleaned_train_data[col_name].nunique() < 10
                             and self.cleaned_train_data[col_name].dtype == "object"]
        self.num_cols_list = [col_name for col_name in self.cleaned_train_data.columns
                              if self.cleaned_train_data[col_name].dtype in ['int64', 'float64']]

        self.__temp_df_list = []
        self.__static_method_list = []

        #feature_creation:


if __name__ == '__main__':
    # print(FeatureData.train_X.columns.tolist())
    pass
