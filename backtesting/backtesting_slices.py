from typing import List

import pandas as pd


class BacktestingSlices:

    def __init__(self, training_length: int = 5, forecasting_length: int = 2, skip_length: int = 2):
        self.training_length = training_length
        self.forecasting_length = forecasting_length
        self.skip_length = skip_length

    def create_training_end_times(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        The given parameters, training_length, forecasting_length, and skip_length and the training end time gives a
        unique and reproducible representation for a backtesting slice.

        :return: A list of possible training end times
        """
        raise NotImplementedError()

    def materialize_slice(self, df: pd.DataFrame, training_end_time: pd.Timestamp) -> (pd.DataFrame, pd.DataFrame):
        """

        :param df:
        :param training_end_time:
        :return: Returns two DataFrames, train and test. The train DataFrame ends training_end_time and the test DataFrame starts at training_end_time + 1 day
        """
        raise NotImplementedError()
