import plotly.express as px
import pandas as pd
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from scipy.stats import pearsonr
from math import sqrt
import statistics 

whole_data = pd.read_csv("US_data/BigUS/largeUS_pred.csv")
fig = px.scatter_mapbox(whole_data, lat = whole_data.cmaq_x, lon = whole_data.cmaq_y, color = 'pm25_value')

fig.show()
