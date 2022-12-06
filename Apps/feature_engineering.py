# FIXME: Conversion into feature extraction class
from pandas import DataFrame
from protlearn.features import aac, entropy, aaindex1, atc

from data_cleaning import CleanedData


class FeatureData:

    train_X = CleanedData.cleaned_train_data.drop('tm', axis=1)
    train_Y = CleanedData.cleaned_train_data.tm
    test_X = CleanedData.cleaned_test_data
    X_list = [train_X, test_X]

    @staticmethod
    def _drop_prot_seq(df: DataFrame) -> None:
        df.drop('protein_sequence', axis=1, inplace=True)


    @staticmethod
    def _get_entropy(df: DataFrame) -> None:
        df['entropy'] = df.protein_sequence.map(entropy)


    @staticmethod
    def _get_atc(df: DataFrame) -> None:
        atom_list = ['C', 'H', 'N', 'O', 'S']
        bond_list = ['tot_bonds', 'single_bonds', 'double_bonds']
        atc_list, bonds_vector = atc(df['protein_sequence'].to_list())

        for i, j in enumerate(atom_list):
            df[f'aac_{j}'] = DataFrame({f'aac_{j}': atc_list[:, i]})

        for k, l in enumerate(bond_list):
            df[f'aac_{l}'] = DataFrame({f'aac_{l}': bonds_vector[:, k]})


    @staticmethod
    def _get_aaindex1(df: DataFrame) -> None:
        aaindex_list, inds = aaindex1(df['protein_sequence'].to_list())

        for i, j in enumerate(inds):
            df[f'aaindex_{j}'] = DataFrame({f'aaindex_{j}': aaindex_list[:, i]})


    @staticmethod
    def _get_seq_len(df: DataFrame) -> None:

        if 'protein_sequence' in df.columns and 'seq_len' not in df.columns:
            df['seq_len'] = df.protein_sequence.map(lambda x: len(x))

        else:
            raise AttributeError('No protein_sequence column in the DataFrame')


    @staticmethod
    def _get_aac(df: DataFrame) -> None:
        aac_list, aac_base = aac(df.protein_sequence.to_list())

        if len(aac_base) >= 1:
            for i, j in enumerate(aac_base):
                df[f'aac_{j}'] = DataFrame({f'aac_{j}': aac_list[:, i]})

        else:
            raise ValueError('Protein sequance base is zero')


    if X_list:
        for i in X_list:
            # i.reset_index()
            _get_seq_len(i)
            _get_aac(i)
            _get_entropy(i)
            _get_aaindex1(i)
            _get_atc(i)
            _drop_prot_seq(i)

    else:
        raise ValueError('No X variable')


if __name__ == '__main__':
    print(FeatureData.train_X.head())
    print(FeatureData.train_X.isna().sum())
    print(len(FeatureData.train_X.seq_len))
    print(len(FeatureData.train_X.aac_O))
    pass
