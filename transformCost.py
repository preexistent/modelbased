import numpy as np
import pandas

def TransformRow(costRow):
    # B = costMatrix.flatten()
    B = np.sort(costRow)
    for idx, i in enumerate(B):
        costRow[np.where(costRow==i)] = idx
    return costRow

def TransformMatrix(costMatrix):
    costMatrix = np.apply_along_axis(TransformRow, axis=1, arr=costMatrix)
    return costMatrix

distanceMatrices = pandas.read_csv('distanceMatrices.csv',
                                       header=None,
                                       nrows=2000,
                                       sep=' ',
                                       dtype='float')
distanceMatrices = distanceMatrices.values
distanceMatrices = TransformMatrix(distanceMatrices)

with open('distanceMatricesDiscrete.csv','ab') as f:
		np.savetxt(f, 
				   distanceMatrices,
				   newline='\n',
				   fmt='%f')

print('Transformation Complete!')
