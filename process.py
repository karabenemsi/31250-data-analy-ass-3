import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


dataset = pd.read_csv('TrainingSet.csv');

# Clean


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
dataset = sel.fit_transform(dataset)
print(dataset.isna().any())
