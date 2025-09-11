from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x, y = fetch_openml('mnist_784', version=1, return_x_y=True, as_frame=False)
x = x / 255.0
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mlp = MLPClassifier (hidden_layer_sizes=(128, 64), max_iter=20, solver="adam", learning_rate_init=0.001)

mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
acc = accuracy_score(y_test, y_pred)