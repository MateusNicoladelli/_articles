# %% md

# NAME

# %%

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import boxcox, pearsonr

# %%

NUM_OF_COLUMNS = 1000
NUM_OF_ROWS = 100
THRESHOLD = 0.05

# %%

data = pd.DataFrame(np.random.rand(NUM_OF_ROWS, NUM_OF_COLUMNS),
                    columns = ['feature_' + str(value) for value in range(1, NUM_OF_COLUMNS + 1)])

data = data.mask(np.random.random(data.shape) < .025)

# %%

data.head()

# %%

columnValues = {}
maskValues = {}
for column in list(data):
	columnValues[column] = data[column].values
	
	mask = ''.join(['1' if value else '0' for value in pd.isnull(data[column])])
	
	if mask not in maskValues.keys():
		maskValues[mask] = [column]
	
	else:
		maskValues[mask].append(column)

# %%

maskValues

# %%

columnsToRemove = [(key, value[0]) for key, value in maskValues.items() if len(value) == 1]

# %%

len(columnsToRemove)

# %%

columnValuesFiltered = {key: value for key, value in columnValues.items() if key not in [value[1] for value in columnsToRemove]}
maskValuesFiltered = {key: value for key, value in maskValues.items() if key not in [value[0] for value in columnsToRemove]}

# %%

for key, value in columnValuesFiltered.items():
	minValue = np.nanmin(value)
	value = [subValue + np.abs(minValue) + 1 for subValue in value if np.isfinite(subValue)]
	value = boxcox(value)[0]
	
	columnValuesFiltered[key] = value

# %%

columnsToUse = []
for _index, (_, _mask) in enumerate(maskValuesFiltered.items()):
	columnValuesFilteredToProcess = {key: value for key, value in columnValuesFiltered.items() if key in _mask}
	
	print(_index, len(maskValuesFiltered.items()), len(columnValuesFilteredToProcess.items()))
	
	distanceMatrix = np.empty((len(columnValuesFilteredToProcess), len(columnValuesFilteredToProcess),))
	distanceMatrix[:] = np.nan
	
	for index1, (filter1, item1) in enumerate(columnValuesFilteredToProcess.items()):
		
		for index2, (filter2, item2) in enumerate(columnValuesFilteredToProcess.items()):
			
			if index1 > index2:
				continue
			
			elif index1 == index2:
				continue
			
			try:
				metric = pearsonr(item1, item2)[0]
			
			except:
				print(list(zip(item1, item2)))
			
			distanceMatrix[index1][index2] = metric
	
	distanceMatrix = np.array(distanceMatrix)
	
	maxIndices = np.argwhere(distanceMatrix >= THRESHOLD)
	
	if len(maxIndices) != 0:
		
		while True:
			indicesFlatten = maxIndices.flatten()
			maxIndex = Counter(indicesFlatten).most_common(1)[0][0]
			
			distanceMatrix[maxIndex, :] = np.nan
			distanceMatrix[:, maxIndex] = np.nan
			
			maxIndices = np.array([value for value in maxIndices if value[0] != maxIndex and value[1] != maxIndex])
			columnValuesFilteredToProcess = {key: value for index, (key, value) in enumerate(columnValuesFilteredToProcess.items()) if index != maxIndex}
			
			if len(maxIndices) == 0:
				break
	
	columnsToUse.extend(list(columnValuesFilteredToProcess.keys()))
	print(len(columnsToUse))

# %%

convertedDFAggregatedFiltered = convertedDFAggregated[[value for value in list(convertedDFAggregated)
                                                       if value in binaryColumns or
                                                       value in categoricalColumns or
                                                       value in columnsToUse or
                                                       value in columnsNotUsed or
                                                       value in ['filter', 'target', 'index', 'train_test', 'monthIndex']]]

# %%

convertedDFAggregatedFiltered.values.shape

# %%
