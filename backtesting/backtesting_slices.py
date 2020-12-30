import collections
from typing import List

import pandas as pd

ONE_DAY = pd.Timedelta(value=1, unit="day")


class BacktestingSlices:

    def __init__(self, training_length: int = 5, forecasting_length: int = 2, skip_length: int = 2):
        self.training_length = training_length
        self.forecasting_length = forecasting_length
        self.skip_length = skip_length

    @property
    def required_length(self):
        return self.training_length + self.forecasting_length

    def iter_contiguous_slices(self, df: pd.DataFrame):
        """
        Yields contiguous slices within each set.
        """
        required_length = self.required_length
        slices = collections.defaultdict(lambda: dict(first=None, last=None, length=0))
        for k, _ in df.sort_values(by=["key", "index"]).iterrows():
            s = slices[k[0]]
            if s["length"] == 0:
                s["first"] = k[1]
                s["last"] = k[1]
                s["length"] = 1
            elif k[1] - s["last"] == ONE_DAY:
                s["last"] = k[1]
                s["length"] += 1
            else:
                if s["length"] >= required_length:
                    yield dict(key=k[0], **s)
                s["first"] = k[1]
                s["last"] = k[1]
                s["length"] = 1

        for sk, s in slices.items():
            if s["length"] >= required_length:
                yield dict(key=sk, **s)

    def create_training_end_times(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        The given parameters, training_length, forecasting_length, and skip_length and the training end time gives a
        unique and reproducible representation for a backtesting slice.

        :return: A list of possible training end times
        """

        end_times = set()

        for s in self.iter_contiguous_slices(df):
            for ts in pd.date_range(
                start=s["first"] + pd.Timedelta(value=self.training_length - 1, unit="day"),
                end=s["last"] - pd.Timedelta(value=self.forecasting_length, unit="day"),
                freq="1D",
            ):
                end_times.add(ts)

        return sorted(end_times)

    def materialize_slice(self, df: pd.DataFrame, training_end_time: pd.Timestamp) -> (pd.DataFrame, pd.DataFrame):
        """

        :param df:
        :param training_end_time:
        :return: Returns two DataFrames, train and test.
        The train DataFrame ends training_end_time and the test DataFrame starts at training_end_time + 1 day
        """
        raise NotImplementedError()
