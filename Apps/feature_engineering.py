from pandas import DataFrame, concat
from protlearn.features import aac, entropy, aaindex1, atc
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from typing import List, AnyStr, Any
from numpy import ndarray, array, concatenate, expand_dims, nanmean, where, isnan, take

from Apps.data_cleaning import CleanedData


class FeatureData(CleanedData):

    __ami_symbols = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    __seq_col = 'protein_sequence'
    __target_column = 'tm'

    @staticmethod
    def _ftn1_fill_na_mean(df: DataFrame, *args: Any, col_name: AnyStr = 'pH') -> None:
        df[col_name].fillna(float(df[col_name].mean()), inplace=True)

    @staticmethod
    def _ftn2_convert_num_columns(df: DataFrame, col_names: List[AnyStr], *args: Any) -> ndarray:
        if len(col_names) > 0:
            return df[col_names].to_numpy()

    @staticmethod
    def _ftc1_get_seq_length(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        return expand_dims(df[col_name].map(lambda x: len(x)).to_numpy(), 1)

    # @staticmethod
    # def _ftc2_get_count_symb(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col,
    #                          amino_res_list: List[AnyStr] = __ami_symbols) -> ndarray:
    #     return array([df[col_name].map(lambda x: x.count(amino_symbol)).tolist() for amino_symbol in amino_res_list]).T

    @staticmethod
    def _ftc3_get_aac(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        aac_list, aac_base = aac(df[col_name].to_list(), method='relative')
        return aac_list if len(aac_base) > 0 else ValueError('Protein sequance base is zero')

    @staticmethod
    def _ftc4_get_atc_atom_arr(df: DataFrame,  *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        atc_arr, bonds_arr = atc(df[col_name].to_list(), method='relative')
        return atc_arr if atc_arr is not None else ValueError('Protein sequance base is zero')

    @staticmethod
    def _ftc4_get_atc_bonds_arr(df: DataFrame,  *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        atc_arr, bonds_arr = atc(df[col_name].to_list(), method='relative')
        return bonds_arr if bonds_arr is not None else ValueError('Protein sequance base is zero')

    # @staticmethod
    # def _ftc5_get_aaindex(df: DataFrame,  *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
    #     aaindex_arr, inds = aaindex1(df[col_name].to_list())
    #     return aaindex_arr if inds is not None else ValueError('Protein sequance base is zero')

    @staticmethod
    def _ftc6_get_entropy(df: DataFrame, *args: Any, col_name: AnyStr = __seq_col) -> ndarray:
        return entropy(df[col_name].to_list(), standardize='none')


    def _join_arr(self, func_list: List[AnyStr], df: DataFrame, *args: Any) -> ndarray:
        func_val_list = []
        col_desc_list = []

        for function in func_list:
            funct_value = getattr(self, function)(df, self.num_cols_list)
            if funct_value is not None:
                func_val_list.append(funct_value)
                if function.startswith('_ftn1') is False:
                    for iterator in range(funct_value.shape[1]):
                        col_desc_list.append(f'{function.split("_")[1]}_{iterator}')

        return concatenate(func_val_list, axis=1), col_desc_list


    def __init__(self):
        super(FeatureData, self).__init__()

        self.data_dict = {'original_train_indices': self.cleaned_train_data.index.tolist(),
                          'original_test_indices': self.cleaned_test_data.index.tolist()}
        self.cleaned_train_data.reset_index(inplace=True)
        self.cleaned_test_data.reset_index(inplace=True)
        self.data_dict['new_train_indices'] = self.cleaned_train_data.index
        self.data_dict['new_test_indices'] = self.cleaned_test_data.index
        self.data_dict['train_y'] = self.cleaned_train_data[self.__target_column]
        self.cleaned_train_data.drop(self.__target_column, axis=1, inplace=True)
        self.data_dict['combined_data'] = concat([self.cleaned_train_data, self.cleaned_test_data])

        # for df_purpose in ['train', 'test']:
        #     for df_type in ['Y', 'X']:
        #         df_name = f'{df_purpose}_{df_type}'
        #
        #         if df_purpose == 'train' and df_type == 'Y':
        #             self.df_name = getattr(self, f'cleaned_{df_purpose}_data')[self.__target_column].to_numpy
        #
        #         elif df_purpose == 'train':
        #             self.df_name = getattr(self, f'cleaned_{df_purpose}_data').drop(self.__target_column, axis=1, inplace=True)

        self.cat_col_list = [col_name for col_name in self.cleaned_train_data.columns
                             if self.cleaned_train_data[col_name].nunique() < 10
                             and self.cleaned_train_data[col_name].dtype == "object"]

        self.num_cols_list = [col_name for col_name in self.cleaned_train_data.columns
                              if self.cleaned_train_data[col_name].dtype in ['int64', 'float64']
                              and col_name.find('id') == -1]

        self.__static_method_list = []

        for method in dir(self):
            if callable(getattr(self, method)) and method.startswith('_ft'):
                self.__static_method_list.append(method)

        self.data_dict['featured_data'], self.data_dict['col_labels'] = self._join_arr(self.__static_method_list, self.data_dict['combined_data'])
        self.data_dict['train_X'] = self.data_dict['featured_data'][self.data_dict['new_train_indices'], :]
        self.data_dict['test_X'] = self.data_dict['featured_data'][self.data_dict['new_test_indices'], :]
        self.data_dict.pop('combined_data')
        self.data_dict.pop('featured_data')
        self.data_dict.pop('new_train_indices')
        self.data_dict.pop('new_test_indices')


if __name__ == '__main__':
    Feature_data = FeatureData()
    print(Feature_data.data_dict['train_X'].shape)
    print(Feature_data.data_dict['train_X'][:10])
    print(Feature_data.data_dict['test_X'].shape)
    print(Feature_data.data_dict.keys())
    print(Feature_data.data_dict['col_labels'])
    pass
