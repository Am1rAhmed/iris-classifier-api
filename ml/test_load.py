import joblib

model = joblib.load("ml/saved_model/model.joblib")

sample = [[7.1, 3.5, 4.4, 0.2]]  

pred = model.predict(sample)
species = ["setosa", "versicolor", "virginica"]
print(f"Predicted class: {pred[0]} -> {species[pred[0]]}")