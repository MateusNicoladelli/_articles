import numpy as np
import random
from copy import deepcopy
from dtaidistance import dtw
from scipy import interpolate

NUM_OF_TRAJECTORIES = 100
MIN_LEN_OF_TRAJECTORY = 10
MAX_LEN_OF_TRAJECTORY = 100

THRESHOLD = 2

# our trajectories will lie in range [-1, 1] with the length of [2, MAX_LEN_OF_ITEMS]

trajectoriesSet = {}

for item in range(NUM_OF_TRAJECTORIES):
	length = random.choice(list(range(MIN_LEN_OF_TRAJECTORY, MAX_LEN_OF_TRAJECTORY + 1)))
	tempTrajectory = np.random.randint(low = -100, high = 100, size = int(length / 5)).astype(float) / 100
	
	oldScale = np.arange(0, int(length / 5))
	interpolationFunction = interpolate.interp1d(oldScale, tempTrajectory)
	
	newScale = np.linspace(0, int(length / 5) - 1, length)
	tempTrajectory = interpolationFunction(newScale)
	
	trajectoriesSet[(str(item),)] = [tempTrajectory]

trajectories = deepcopy(trajectoriesSet)  # we make a copy just for convenience
distanceMatrixDictionary = {}  # to avoid recalculations and speed-up

iteration = 1
while True:  # we will stop when there is nothing to group
	distanceMatrix = np.empty((len(trajectories), len(trajectories),))
	distanceMatrix[:] = np.nan  # np.nan distance matrix
	
	for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
		tempArray = []
		
		for index2, (filter2, trajectory2) in enumerate(trajectories.items()):
			
			if index1 > index2:  # dtw metric is symmetrical
				continue
			
			elif index1 == index2:  # same trajectories => distance is 0
				continue
			
			else:
				unionFilter = filter1 + filter2
				sorted(unionFilter)  # for convenience
				
				if unionFilter in distanceMatrixDictionary.keys():  # we already calculated this distance before
					distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)
					
					continue
				
				metric = []
				for subItem1 in trajectory1:
					
					for subItem2 in trajectory2:
						metric.append(dtw.distance(subItem1, subItem2))
				
				metric = max(metric)  # here we have different options: use max() or mean() or any other custom function
				
				distanceMatrix[index1][index2] = metric
				distanceMatrixDictionary[unionFilter] = metric
	
	minValue = np.min(list(distanceMatrixDictionary.values()))  # we take the lowest distance
	
	if minValue > THRESHOLD:  # if the lowest distance is higher than our threshold => stop (no pairs to group anymore)
		break
	
	minIndices = np.where(distanceMatrix == minValue)  # we find pair(s) that has/ve the lowest distance
	minIndices = list(zip(minIndices[0], minIndices[1]))
	
	minIndex = minIndices[0]  # we work with the only one pair at a time
	
	filter1 = list(trajectories.keys())[minIndex[0]]  # names of the items to group
	filter2 = list(trajectories.keys())[minIndex[1]]
	
	trajectory1 = trajectories.get(filter1)  # values of the items to group
	trajectory2 = trajectories.get(filter2)
	
	unionFilter = filter1 + filter2
	sorted(unionFilter)
	
	# now we group the items in one cluster
	trajectoryGroup = trajectory1 + trajectory2
	
	trajectories = {key: value for key, value in trajectories.items()
	                if all(value not in unionFilter for value in key)}  # while we have group now ([filter1, filter22]) we need to remove single trajectories filter1 and filter2
	
	distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
	                            if all(value not in unionFilter for value in key)}  # we reduce our distanceMatrixDictionary here
	
	trajectories[unionFilter] = trajectoryGroup
	
	print(iteration, 'finished!')
	iteration += 1
	
	if len(list(trajectories.keys())) == 1:
		break

for key, _ in trajectories.items():
	print(key)

for key, value in trajectories.items():
	
	if len(key) == 1:
		continue
	
	figure, axes = plt.subplots(nrows = 1, ncols = 2)
	
	for subValue in value:
		axes[1].plot(subValue)
		
		oldScale = np.arange(0, len(subValue))
		interpolateFunction = interpolate.interp1d(oldScale, subValue)
		
		newScale = np.linspace(0, len(subValue) - 1, MAX_LEN_OF_TRAJECTORY)
		subValue = interpolateFunction(newScale)
		
		axes[0].plot(subValue)
	
	plt.show()
