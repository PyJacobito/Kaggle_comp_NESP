# FIXME: Conversion into feature extraction class
from pandas import DataFrame, concat
from protlearn.features import aac, entropy, aaindex1, atc
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
from lev_dist_func import levenshtein_ratio_and_distance


from data_cleaning import CleanedData


class FeatureData:

    train_X = CleanedData.cleaned_train_data.drop('tm', axis=1)
    train_Y = CleanedData.cleaned_train_data.tm
    test_X = CleanedData.cleaned_test_data
    X_list = [train_X, test_X]
    base_protein_seq = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'

    @staticmethod
    def _get_base_len_dif(df:DataFrame, base_str: str) -> None:
        base_len = len(base_str)
        df['base_len_dif'] = df.protein_sequence.map(lambda x: abs(len(x)-base_len) / base_len)


    @staticmethod
    def _get_dif_aar_base(df:DataFrame, base_str: str) -> None:
        aac_base = 'ACDEFGHIKLMNPQRSTVWY'
        aac_list = [i for i in aac_base]
        df[f'diff_aar_base_all'] = df['protein_sequence'].map(lambda x: 0)

        for i in aac_list:
            base_count = base_str.count(i)
            df[f'diff_aar_base_{i}'] = df['protein_sequence'].map(lambda x: abs(x.count(i)-base_count))
            df[f'diff_aar_base_all'] += df[f'diff_aar_base_{i}']

        for i in aac_list:
            df[f'diff_aar_base_{i}'] = df[f'diff_aar_base_{i}'] / df[f'diff_aar_base_all']
            # df.drop([f'diff_aar_base_{i}'], axis=1, inplace=True)

        # df.drop([f'diff_aar_base_all'], axis=1, inplace=True)


    @staticmethod
    def _get_kmean_cluster(df: DataFrame, col_name: str, cluster_numb: int) -> None:
        reshaped_data = df[col_name].to_numpy().reshape(-1, 1)
        # scaled_data = MinMaxScaler().fit(reshaped_data).transform(reshaped_data)
        scaled_data = reshaped_data
        # normalized_data = Normalizer().fit(scaled_data).transform(scaled_data)
        normalized_data = scaled_data
        kmeans = KMeans(n_clusters=cluster_numb).fit(normalized_data)
        df[f"cluster_{col_name}"] = DataFrame(kmeans.predict(normalized_data), columns=[f"cluster_{col_name}"])
        return kmeans

    @staticmethod
    def _drop_prot_seq(df: DataFrame) -> None:
        df.drop('protein_sequence', axis=1, inplace=True)


    @staticmethod
    def _get_entropy(df: DataFrame) -> None:
        entropy_list = entropy(df.protein_sequence.to_list(), standardize='none')
        df['entropy'] = DataFrame(entropy_list, columns=['entropy'])


    @staticmethod
    def _get_atc(df: DataFrame) -> None:
        atom_list = ['C', 'H', 'N', 'O', 'S']
        bond_list = ['tot_bonds', 'single_bonds', 'double_bonds']
        atc_list, bonds_vector = atc(df['protein_sequence'].to_list(), method='relative')

        # for i, j in enumerate(atom_list):
        #     df[f'aac_{j}'] = DataFrame({f'aac_{j}': atc_list[:, i]})

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
    def _get_lev_ratio(df: DataFrame, base_str: str) -> None:
        df['lev_ratio'] = df['protein_sequence'].map(lambda x: levenshtein_ratio_and_distance(x, base_str))


    @staticmethod
    def _get_common_intersections(df: DataFrame, base_str: str) -> None:
        # lower_index_df = df.index[df['seq_len'] <= 221]
        # above_index_df = df.index[df['seq_len'] > 221]
        #
        # temp_df = DataFrame(index=df.index)

        def _get_first_values(first_str, second_str):
            str_list = []
            i = 0

            if len(first_str) > 221:
                while i <= len(second_str)-1:
                    str_list.append(first_str[i])
                    i += 1
                return str_list

            else:
                while i <= len(first_str) - 1:
                    str_list.append(first_str[i])
                    i += 1
                return str_list

        for i in range(len(base_str)):
            sequances_list = df['protein_sequence'].map(lambda x: _get_first_values(x, base_str)).tolist()


        temp_df = DataFrame(sequances_list).apply(LabelEncoder().fit_transform)
        print(temp_df.head())
        df[temp_df.columns] = temp_df




        # temp_df = df[df['seq_len'] <= 221]
        # sequences = [list(string) for string in df[lower_index_df].values.tolist()]
        # sequences_test = pd.DataFrame(sequences)
        # sequences_test = sequences_test.apply(LabelEncoder().fit_transform)


    @staticmethod
    def _get_aac(df: DataFrame) -> None:
        aac_list, aac_base = aac(df.protein_sequence.to_list(), method='relative')

        if len(aac_base) >= 1:
            for i, j in enumerate(aac_base):
                df[f'aac_{j}'] = DataFrame({f'aac_{j}': aac_list[:, i]})

        else:
            raise ValueError('Protein sequance base is zero')


    if X_list:
        for i in X_list:
            i.reset_index(inplace=True)
            _get_seq_len(i)
            _get_base_len_dif(i, base_protein_seq)
            # _get_lev_ratio(i, base_protein_seq)
            # _get_common_intersections(i, base_protein_seq)
            _get_dif_aar_base(i, base_protein_seq)
            _get_aac(i)
            _get_entropy(i)

            if 'cluster_pH' not in locals():
                cluster_pH = _get_kmean_cluster(i, 'pH', 4)

            else:
                i["cluster_pH"] = DataFrame(cluster_pH.predict(i['pH'].to_numpy().reshape(-1,1)), columns=['cluster_pH'])

            # if 'cluster_len' not in locals():
            #     cluster_len = _get_kmean_cluster(i, 'seq_len', 50)
            #
            # else:
            #     i["cluster_seq_len"] = cluster_len.predict(i['seq_len'].to_numpy().reshape(-1,1))

            _get_aaindex1(i)
            # _get_atc(i)
            _drop_prot_seq(i)

    else:
        raise ValueError('No X variable')


if __name__ == '__main__':
    # print(FeatureData.train_X.head())
    print(FeatureData.train_X.columns.tolist())
    # print(FeatureData.train_X.nunique())
    # print(FeatureData.train_X.isna().sum())
    # print(FeatureData.train_X['seq_len'][FeatureData.train_X['seq_len'] == 221].count())
    # print(FeatureData.train_X['seq_len'].count(221))
    # print(FeatureData.train_X.entropy.head())
    # print(len(FeatureData.train_X.seq_len))
    # print(len(FeatureData.train_X.aac_O))
    # FeatureData.train_X.aac_A.to_csv("aac_A.csv")
    # print(FeatureData.train_X.loc[26303, ["protein_sequence", 'aac_A']])
    pass
