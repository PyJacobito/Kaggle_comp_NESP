from data_loading import LoadedData


class CleanedData:

    @classmethod
    def _remove_irrelevant_columns(self):
        for iter1 in LoadedData.test_data.columns.to_list():
            if LoadedData.train_data[iter1].isin(LoadedData.test_data[iter1].unique()).sum() == 0\
                    and iter1 != 'protein_sequence':
                if 'cleaned_train_data' not in locals():
                    self.cleaned_train_data = LoadedData.train_data.drop(iter1, axis=1, )
                    self.cleaned_test_data = LoadedData.test_data.drop(iter1, axis=1)

                else:
                    self.cleaned_train_data.drop(iter1, axis=1, inplace=True)
                    self.cleaned_test_data.drop(iter1, axis=1, inplace=True)

        return self.cleaned_train_data, self.cleaned_test_data


if __name__ == '__main__':
    print(LoadedData.train_data.data_source.isin(['Novozymes']).sum())
    print("\n")
    print("\n")
    print(CleanedData._remove_irrelevant_columns())
