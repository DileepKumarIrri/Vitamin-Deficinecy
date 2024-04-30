from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define base models
model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()

# Create a voting ensemble
ensemble = VotingClassifier(estimators=[('lr', model1), ('svm', model2), ('rf', model3)], voting='hard')

# Train the ensemble
ensemble.fit(train_data, train_labels)

# Evaluate the ensemble
accuracy = ensemble.score(test_data, test_labels)
print("Ensemble Accuracy:", accuracy)
