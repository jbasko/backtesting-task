import pandas as pd
import numpy as np

class ArtificialDatasets(object):
    @classmethod
    def _create_series(self, length=10, start_date="2017-01-01", freq="1D"):
        df = pd.DataFrame(pd.date_range(start_date, periods=length, freq=freq, name="index"))
        df["target"] = np.arange(length)
        return df

    @classmethod
    def create_single_key_timeseries(cls, start_date="2017-01-01", length=10, key="A") -> pd.DataFrame:
        df = cls._create_series(length=length, start_date=start_date)
        df["key"] = key
        return df

    @classmethod
    def create_multiple_timeseriesset(cls):
        df_1 = cls.create_single_key_timeseries(start_date="2017-01-01", length=10, key="A")
        df_2 = cls.create_single_key_timeseries(start_date="2017-01-01", length=8, key="B")
        df_3 = cls.create_single_key_timeseries(start_date="2017-01-03", length=8, key="C")
        df_4 = cls.create_single_key_timeseries(start_date="2017-01-02", length=8, key="D")
        df_5 = cls.create_single_key_timeseries(start_date="2017-01-04", length=7, key="E")
        df = pd.concat([df_1, df_2, df_3, df_4, df_5])
        return df.set_index(["key", "index"])