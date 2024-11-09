import numpy as np

# Given list of probabilities
probabilities = [
    9.719550609588623047e-01, 7.565616369247436523e-01, 9.174848198890686035e-01,
    4.920566976070404053e-01, 3.085645735263824463e-01, 5.866732001304626465e-01,
    3.890978097915649414e-01, 1.821730285882949829e-01, 5.726953744888305664e-01,
    1.151409298181533813e-01, 3.688287436962127686e-01, 2.504866421222686768e-01
]

# Convert the list to a numpy array
probabilities_array = np.array(probabilities)

# Find the index of the maximum value
max_index = np.argmax(probabilities_array)

print("Index of the maximum probability:", max_index)
