import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

st.title("AI and Machine Learning Integrated Projects")

# Interactive KNN Classifier
st.header("KNN Classifier Decision Boundary")
data = load_iris()
X = data.data[:, :2]
y = data.target

# User input for KNN parameters
test_size = st.slider("Test size (fraction of dataset)", 0.1, 0.5, 0.3)
n_neighbors = st.slider("Number of neighbors (K)", 1, 15, 5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
ax.set_xlabel(data.feature_names[0])
ax.set_ylabel(data.feature_names[1])
ax.set_title("KNN Classifier Decision Boundary")
st.pyplot(fig)

# Interactive Mandelbrot Set
st.header("Mandelbrot Set Visualization")

# User inputs for Mandelbrot set parameters
width = st.slider("Width of the image (pixels)", 100, 1000, 800)
height = st.slider("Height of the image (pixels)", 100, 1000, 800)
xmin, xmax = st.slider("X-axis range", -4.0, 4.0, (-2.5, 1.5))
ymin, ymax = st.slider("Y-axis range", -4.0, 4.0, (-2.0, 2.0))
max_iter = st.slider("Maximum iterations", 100, 2000, 1000)

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z**2 + c
        n += 1
    return n

image = np.zeros((height, width))
for x in range(width):
    for y in range(height):
        real = xmin + (x / width) * (xmax - xmin)
        imag = ymin + (y / height) * (ymax - ymin)
        c = complex(real, imag)
        m = mandelbrot(c, max_iter)
        image[y, x] = m

fig, ax = plt.subplots()
ax.imshow(image, extent=(xmin, xmax, ymin, ymax), cmap="hot")
ax.set_title("Mandelbrot Set")
st.pyplot(fig)

# Interactive Population Growth Model
st.header("Population Growth Model")

# User inputs for logistic growth parameters
P0 = st.number_input("Initial population (P0)", 10, 1000, 100)
r = st.slider("Growth rate (r)", 0.01, 1.0, 0.1)
K = st.number_input("Carrying capacity (K)", 100, 10000, 1000)
time_span = st.slider("Time span for simulation", 10, 500, 100)

def logistic_growth(P, r, K, t):
    return (K * P) / (P + (K - P) * np.exp(-r * t))

time = np.linspace(0, time_span, 500)
population = logistic_growth(P0, r, K, time)

fig, ax = plt.subplots()
ax.plot(time, population, label="Logistic Growth")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.set_title("Population Growth Over Time")
ax.legend()
ax.grid()
st.pyplot(fig)
