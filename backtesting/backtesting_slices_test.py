import unittest

from backtesting.artificial_datasets import ArtificialDatasets
from backtesting.backtesting_slices import BacktestingSlices

class TestBacktestingSlices(unittest.TestCase):
    @unittest.skip
    def test_create_training_end_times(self):
        df = ArtificialDatasets.create_multiple_timeseriesset()
        bt_slice = BacktestingSlices(training_length=5, forecasting_length=3, skip_length=1)
        training_end_times = bt_slice.create_training_end_times(df)
        self.assertEqual(len(training_end_times), 3)
        self.assertEqual(training_end_times, ["2017-01-05", "2017-01-06", "2017-01-07"])

    @unittest.skip
    def test_materialize_slice(self):
        df = ArtificialDatasets.create_multiple_timeseriesset()
        bt_slice = BacktestingSlices(training_length=5, forecasting_length=3, skip_length=1)
        training_end_times = bt_slice.create_training_end_times(df)
        train_slice_1, test_slice_1 = bt_slice.materialize_slice(df, training_end_times[0])
        keys = train_slice_1.reset_index()['key'].drop_duplicates()
        self.assertEqual(len(train_slice_1) + len(test_slice_1), 16)
        self.assertEqual(keys, ['A', 'B'])

        train_slice_2, test_slice_2 = bt_slice.materialize_slice(df, training_end_times[1])
        keys = train_slice_2.reset_index()['key'].drop_duplicates()
        self.assertEqual(len(train_slice_2) + len(test_slice_2), 16)
        self.assertEqual(keys, ['A', 'D'])

        train_slice_3, test_slice_3 = bt_slice.materialize_slice(df, training_end_times[2])
        keys = train_slice_1.reset_index()['key'].drop_duplicates()
        self.assertEqual(len(train_slice_3) + len(test_slice_3), 16)
        self.assertEqual(keys, ['A', 'C'])
