from sklearn.datasets import load_iris

iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer
import mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
X_iris = iris_dataset.data
Y_iris = iris_dataset.target
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
X = X_iris[:, :2]
Y = Y_iris


def Draw1():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    colors = ['red', 'pink', 'green'];
    for i in range(len(colors)):
        xs = X_train[:, 0][y_train == i]
        ys = X_train[:, 1][y_train == i]
        plt.scatter(xs, ys, c=colors[i])
    plt.legend(iris_dataset.target_names)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal wwidth')
    plt.show();


def Draw2():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)


def DuDoanLoaiHoa():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)  # Huấn luyện mô hình trên dữ liệu huấn luyện

    # Dự đoán trên một mẫu dữ liệu mới từ dữ liệu thực tế
    X_new = np.array([[5, 2.9]])  # Dữ liệu thực tế từ tập dữ liệu Iris
    X_new_scaled = scaler.transform(X_new)  # Scale dữ liệu mới theo cùng phương pháp đã sử dụng cho dữ liệu huấn luyện
    prediction = knn.predict(X_new_scaled)  # Dự đoán nhãn cho dữ liệu mới
    print("Predicted target name for new data: {}".format(iris_dataset['target_names'][prediction]))

    # Dự đoán trên tập dữ liệu kiểm tra
    y_pred = knn.predict(X_test)
    print("Test set predictions:\n {}".format(y_pred))


def Draw3():
    # generate dataset
    X, y = mglearn.datasets.make_forge()
    # plot dataset
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    print("X.shape: {}".format(X.shape))
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()


def Draw4():
    cancer = load_breast_cancer()
    print("cancer.keys(): \n{}".format(cancer.keys()))
    print("Shape of cancer data: {}".format(cancer.data.shape))
    print("Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
    print("Feature names:\n{}".format(cancer.feature_names))

    X, y = mglearn.datasets.make_forge()
    mglearn.plots.plot_knn_classification(n_neighbors=1, X_train=X, y_train=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    print("Test set predictions: {}".format(clf.predict(X_test)))


def Draw5():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
    training_accuracy = []
    test_accuracy = []
    # try n_neighbors from 1 to 10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # build the model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()

def Draw6():
    X, y = mglearn.datasets.make_wave(n_samples=40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)

    # instantiate the model and set the number of neighbors to consider to 3
    reg = KNeighborsRegressor(n_neighbors=3)
    # fit the model using the training data and training targets
    reg.fit(X_train, y_train)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # create 1,000 data points, evenly spaced between -3 and 3
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # make predictions using 1, 3, or 9 neighbors
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
        ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                n_neighbors, reg.score(X_train, y_train),
                reg.score(X_test, y_test)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target",
                    "Test data/target"], loc="best")
    plt.show()

    print("lr.coef_: {}".format(lr.coef_))
    print("lr.intercept_: {}".format(lr.intercept_))

    ridge = Ridge().fit(X_train, y_train)
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

def Draw7():
    X, y = mglearn.datasets.make_wave(n_samples=40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    ridge = Ridge().fit(X_train, y_train)

    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
    plt.plot(lr.coef_, 'o', label="LinearRegression")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.ylim(-25, 25)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Draw1()
    # Draw2()
    # DuDoanLoaiHoa()
    # Draw3()
    #Draw5()
    #Draw6()
    Draw7()

    # iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # # create a scatter matrix from the dataframe, color by y_train
    # grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
    #                         hist_kwds={'bins': 20}, s=60, alpha=.8)

    # print("X_iris shape: ",X_iris.shape)
    # print("Y_iris shape: ", Y_iris.shape)
    #
    # print("X_train shape: {}".format(X_train.shape))
    # print("y_train shape: {}".format(y_train.shape))
    #
    # print("X_test shape: {}".format(X_test.shape))
    # print("y_test shape: {}".format(y_test.shape))

    # print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
    # print(iris_dataset['DESCR'][:193] + "\n...")
    # print("Target names: {}".format(iris_dataset['target_names']))
    # print("Feature names: \n{}".format(iris_dataset['feature_names']))
    # print("Type of data: {}".format(type(iris_dataset['data'])))
