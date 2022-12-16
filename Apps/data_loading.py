from os import path, sep, pardir, getcwd, listdir
from typing import AnyStr, List, Dict, Set, Any

from pandas import DataFrame, read_csv, concat


class LoadedData:
    """Class storing the train and test data"""

    @staticmethod
    def _get_loading_paths(searched_tags: Set[AnyStr] = ('train', 'test', 'train_updates')) -> Dict[AnyStr, AnyStr]:
        """Function returns the loading paths for the relevant data"""


        def _join_list_to_str(*args: List[Any]) -> str:
            """Function joins the first list arguments"""
            str_var = ""

            for arg in args:
                if arg != args[-1]:
                    str_var += arg[0] + '\\'

                else:
                    str_var += arg[0]

            return str_var


        def _check_list(list_var: List[AnyStr]) -> bool:
            """Function checks the given list length"""
            return True if len(list_var) == 1 else False


        def _get_working_directory(str_var: AnyStr = "") -> AnyStr:
            """Function returns the path to data in the working directory"""
            return str(path.normpath(getcwd() + sep + pardir)) + f'\\{str_var}'


        def _get_all_subdirectories(path_str: AnyStr = "") -> List[AnyStr]:
            """Function returns all the possible subdirectories"""
            if path_str:
                return listdir(path_str)

            else:
                return listdir(_get_working_directory())

        docs_subdir_list = [subdir for subdir in _get_all_subdirectories() if subdir.lower().find('docs') >= 0]

        if _check_list(docs_subdir_list):
            docs_path = _get_working_directory(_join_list_to_str(docs_subdir_list))
            data_subdir_list = [subdir for subdir in _get_all_subdirectories(docs_path)
                                if subdir.lower().find('data') >= 0]

            if _check_list(data_subdir_list):
                data_path = _get_working_directory(_join_list_to_str(docs_subdir_list, data_subdir_list))
                tag_dict = {}

                for tag in searched_tags:
                    tag_subdir_list = [subdir for subdir in _get_all_subdirectories(data_path)
                                       if subdir.lower().find(f'{tag}') >= 0]

                    if len(tag_subdir_list) > 0:
                        tag_path = _get_working_directory(_join_list_to_str(docs_subdir_list,
                                                                            data_subdir_list,
                                                                            tag_subdir_list))
                        tag_dict[f'{tag}_df'] = tag_path

                return tag_dict

            else:
                raise FileExistsError('No or to much of data directory in the docs')

        else:
            raise FileExistsError('No or to much of docs directory in the current working directory')

    @staticmethod
    def _get_corrected_train_data(train_df: DataFrame, update_df: DataFrame) -> DataFrame:
        """Function returns the corrected train data with the last update file"""
        update_df.data_source = train_df.loc[update_df.index, 'data_source']
        update_indexes = update_df[update_df.protein_sequence.isna()].index
        update_df.loc[update_indexes, 'protein_sequence'] = train_df.loc[update_indexes, 'protein_sequence']
        train_df.drop(update_df.index, inplace=True)
        return concat([train_df, update_df])


    def __init__(self, initialzation: bool = True) -> None:
        if initialzation is True:
            self.loading_paths_dict = self._get_loading_paths()
            self.test_df = read_csv(self.loading_paths_dict['test_df'], index_col="seq_id")
            self.train_df = self._get_corrected_train_data(
                read_csv(self.loading_paths_dict['train_df'], index_col="seq_id"),
                read_csv(self.loading_paths_dict['train_updates_df'], index_col="seq_id")
            )

            del self.loading_paths_dict


if __name__ == "__main__":
    Loaded_data = LoadedData()
    print(Loaded_data.train_df.head(5))
    pass
