import pytest

from backtesting.artificial_datasets import ArtificialDatasets
from backtesting.backtesting_slices import BacktestingSlices


@pytest.fixture
def mts_set():
    return ArtificialDatasets.create_multiple_timeseriesset()


def test_create_training_end_times(mts_set):
    df = mts_set
    bt_slice = BacktestingSlices(training_length=5, forecasting_length=3, skip_length=1)
    training_end_times = bt_slice.create_training_end_times(df)
    assert len(training_end_times) == 3
    assert training_end_times == ["2017-01-05", "2017-01-06", "2017-01-07"]


def test_materialize_slice(mts_set):
    df = mts_set
    bt_slice = BacktestingSlices(training_length=5, forecasting_length=3, skip_length=1)
    training_end_times = bt_slice.create_training_end_times(df)
    train_slice_1, test_slice_1 = bt_slice.materialize_slice(df, training_end_times[0])
    keys = train_slice_1.reset_index()["key"].drop_duplicates()
    assert len(train_slice_1) + len(test_slice_1) == 16
    assert keys == ["A", "B"]

    train_slice_2, test_slice_2 = bt_slice.materialize_slice(df, training_end_times[1])
    keys = train_slice_2.reset_index()["key"].drop_duplicates()
    assert len(train_slice_2) + len(test_slice_2) == 16
    assert keys == ["A", "D"]

    train_slice_3, test_slice_3 = bt_slice.materialize_slice(df, training_end_times[2])
    keys = train_slice_1.reset_index()["key"].drop_duplicates()
    assert len(train_slice_3) + len(test_slice_3) == 16
    assert keys == ["A", "C"]
