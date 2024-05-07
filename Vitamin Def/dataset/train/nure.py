import numpy as np
import skfuzzy as fuzz

# Define fuzzy sets for classes based on probability ranges
# You may need to adjust these ranges based on your specific problem
probabilities = [
    9.719550609588623047e-01, 7.565616369247436523e-01, 9.174848198890686035e-01,
    4.920566976070404053e-01, 3.085645735263824463e-01, 5.866732001304626465e-01,
    3.890978097915649414e-01, 1.821730285882949829e-01, 5.726953744888305664e-01,
    1.151409298181533813e-01, 3.688287436962127686e-01, 2.504866421222686768e-01
]

# Define fuzzy sets for classes
classes = np.arange(1, 13, 1)

# Define fuzzy membership functions for classes
# You may need to adjust these membership functions based on your specific problem
class_mfs = []
for i in range(12):
    class_mfs.append(fuzz.trimf(classes, [i + 0.5, i + 1, i + 1.5]))

# Calculate membership degrees for each class based on probabilities
membership_degrees = []
for i in range(12):
    membership_degrees.append(fuzz.interp_membership(classes, class_mfs[i], probabilities[i]))

# Define and train a neural network to refine fuzzy predictions (optional)
# You can use any neural network architecture and training method suitable for your problem

# Combine fuzzy predictions with neural network predictions to get final classification
# You may need to adjust this combination method based on your specific problem
final_classification = np.argmax(membership_degrees) + 1  # Adding 1 to convert index to class label

print("Final classification:", final_classification)
