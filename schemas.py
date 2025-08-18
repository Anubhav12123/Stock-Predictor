import pandera as pa
from pandera import Column
from pandera.typing import Series
import pandas as pd

class OHLCVSchema(pa.DataFrameModel):
    Date: Series[pd.Timestamp] = Column(pa.DateTime, coerce=True)
    Open: Series[float]
    High: Series[float]
    Low: Series[float]
    Close: Series[float]
    Volume: Series[int]
    # optional columns
    AdjClose: Series[float] | None
    Ticker: Series[str]

    class Config:
        coerce = True
