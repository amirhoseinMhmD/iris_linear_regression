import pandas as pd
import numpy as np

dictionary = {
    1: "Iris-setosa",
    2: 'Iris-versicolor',
    3: 'Iris-virginica'
}


def separate(dataset):
    dataset_size = len(dataset)
    class_size = dataset_size // 3
    map = {
        1: dataset[:class_size],
        2: dataset[class_size: 2 * class_size],
        3: dataset[2 * class_size:]
    }
    return map


def generate_matrix(df):
    t = df["petal_length"].to_numpy()[:40]
    f0, f1, f2, f4 = df['type'], df['sepal_length'], df['sepal_width'], \
                     df['petal_width']
    x = pd.DataFrame(f0).join(f1).join(f2).join(f4).to_numpy()[:40]
    return t, x


def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


def test(test_data, w):
    error = 0
    for i in range(len(test_data)):
        result = w[0] + w[1] * test_data.iloc[i]['sepal_length'] + w[2] * test_data.iloc[i]['sepal_width'] + w[3] * \
                 test_data.iloc[i]['petal_width']
        error += 0.5 * (result - test_data.iloc[i]['petal_length']) ** 2
    return error


def linear_regression():
    dataset = load_data("iris_dataset.csv")
    separate_dataset = separate(dataset)

    for i in dictionary.keys():
        print(dictionary[i])
        separate_dataset[i] = separate_dataset[i].replace([dictionary[i]], i)
        t, x = generate_matrix(separate_dataset[i])

        test_data = separate_dataset[i][40:]

        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), t)
        print(w)
        error = test(test_data, w)
        print("Error : ", error)
        print("--------------------------------------------------------------------------------")


if __name__ == '__main__':
    linear_regression()
