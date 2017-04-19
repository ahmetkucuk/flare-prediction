
import pandas as pd
import numpy as np
from flare_dataset import get_data
from configurations import get_norm_func


index = pd.date_range('1/1/2000', periods=60, freq='T')


norm_func = get_norm_func("z_score")

# dataset = get_data(name="12_12", data_root="/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARFinal", norm_func=norm_func, should_augment=True)
# data = dataset.get_all_data()

#series = pd.Series(data[0].T[0], index=index)
series = pd.Series(range(60), index=index)

print(series.shift(periods=2))
#print(len(series.resample('2T').sum()))
#print(len(series.shift('2T').sum()))
#print(len(series.resample('30S', closed='right').bfill()))
