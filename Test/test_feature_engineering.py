from Apps.feature_engineering import FeatureData
from pandas import DataFrame
from numpy import ndarray, array
from numpy.random import rand
from random import seed
from string import ascii_lowercase, ascii_uppercase

seed(123)


# def test_construction():
#     assert FeatureData()
#     assert FeatureData().__getattribute__('_FeatureData__ami_symbols') == ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
#                                                                            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
#                                                                            'W', 'Y']
#     assert FeatureData().__getattribute__('_FeatureData__seq_col') == 'protein_sequence'
#     assert FeatureData().__getattribute__('_FeatureData__target_column') == 'tm'

# def test_construction():
#     assert TestClass()


class TestFeatureData:

    temp_narr = rand(10, 10)
    col_names = [ascii_lowercase[i] for i in range(len(temp_narr))]
    temp_df = DataFrame(temp_narr, columns=col_names)
    class_var = FeatureData()
    seq_col = 'protein_sequence'

    def test_construction(self):
        assert self.class_var


    def test_private_variable(self):
        assert self.class_var.__getattribute__('_FeatureData__ami_symbols') == ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                                                                'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
                                                                                'T', 'V', 'W', 'Y']
        assert self.class_var.__getattribute__('_FeatureData__seq_col') == 'protein_sequence'
        assert self.class_var.__getattribute__('_FeatureData__target_column') == 'tm'


    def test_ftn1_convert_num_columns(self):
        try:
            conv_df = self.class_var._ftn1_convert_num_columns(self.temp_df, self.col_names)

        except Exception:
            assert False

        assert isinstance(conv_df, ndarray)
        assert conv_df.all() == self.temp_narr.all()

        try:
            assert self.class_var._ftn1_convert_num_columns()
            assert self.class_var._ftn1_convert_num_columns(self.temp_df)
            assert self.class_var._ftn1_convert_num_columns(self.temp_narr, self.col_names)
            assert self.class_var._ftn1_convert_num_columns(self.col_names, self.col_names)

        except TypeError:
            assert True

        except IndexError:
            assert True


    def test_ftc1_get_seq_length(self):
        assert True