from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the iris dataset
iris = load_iris()

print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y,
 random_state = 42,
stratify = y,
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
joblib.dump(model, 'iris_model.pkl')
predicted_species = model.predict([[5.1, 3.5, 1.4, 0.2]])
flower_name = iris.target_names[predicted_species][0]
print("Predicted flower species:", flower_name)