import pandas as pd
import numpy as np

data_type = {
        'category': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 41, 42, 43, 44, 45, 46, 47,
                     48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59],
        'float': [17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 53],
        'integer': [1, 14, 15, 16, 18, 19, 20]

    }


def read_csv_data_(_data_path):

    f = open(_data_path, 'r')
    lines = f.readlines()

    extracted_data = []
    class_label = []
    for line in lines:
        split_line = line.split(',')

        if not lines.index(line):
            continue

        if 'NA' in split_line:
            continue

        for i, element in enumerate(split_line):
            if i in data_type['category']:
                split_line[i] = int(element)

            elif i in data_type['float']:
                split_line[i] = float(element)

            else:
                split_line[i] = int(element)

        extracted_data.append(split_line[0:59])
        class_label.append(split_line[59])
    f.close()

    extracted_data = np.array(extracted_data)

    return extracted_data, class_label


def read_csv_to_data_frame(_data_path):

    return pd.read_csv(_data_path)


##############################################################
#                         Classifier                         #
##############################################################
def logistic_regression_(_train_data, _train_label, _test_data):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression().fit(_train_data, _train_label).predict(_test_data)


def naive_bayes_(_train_data, _train_label, _test_data):
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB().fit(_train_data, _train_label).predict(_test_data)


def decision_tree_(_train_data, _train_label, _test_data):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier().fit(_train_data, _train_label).predict(_test_data)


def random_forest_(_train_data, _train_label, _test_data):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier().fit(_train_data, _train_label).predict(_test_data)


def ada_boost_(_train_data, _train_label, _test_data):
    from sklearn.ensemble import AdaBoostClassifier

    return AdaBoostClassifier().fit(_train_data, _train_label).predict(_test_data)


##############################################################
#                  One-Hot Encode related                    #
##############################################################
def one_hot_encode(_data):
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder().fit_transform(_data)


##############################################################
#                      train-test split                      #
##############################################################
def split_train_test(_data, _label):
    from sklearn.model_selection import train_test_split

    return train_test_split(_data, _label, test_size=0.33, random_state=72170233)


##############################################################
#                compute performance measure                 #
##############################################################
def compute_result_(_test_label, _predict_label):
    from sklearn import metrics
    # from sklearn.metrics import precision_score, recall_score, accuracy_score

    precision = metrics.precision_score(_test_label, _predict_label)
    recall = metrics.recall_score(_test_label, _predict_label)
    accuracy = metrics.accuracy_score(_test_label, _predict_label)

    fpr, tpr, threshold = metrics.roc_curve(_test_label, _predict_label)
    auc = metrics.auc(fpr, tpr)
    print("Accuracy:", accuracy, "AUC:", auc, "Precision:", precision, "Recall:", recall)


def classify_continuous_to_category(_input_data):

    for i in data_type['float']:
        temp = _input_data[:, i]
        mean = np.mean(temp)
        for j in range(len(temp)):
            if temp[j] > mean:
                _input_data[j, i] = 0
            else:
                _input_data[j, i] = 1

    for i in data_type['integer']:
        temp = _input_data[:, i]
        mean = np.mean(temp)
        for j in range(len(temp)):
            if temp[j] > mean:
                _input_data[j, i] = 0
            else:
                _input_data[j, i] = 1

    return _input_data


def main():
    data_path = 'D:\\workspace\\github\\machine_learning_homework\\MLdata2_R.csv'
    data, label = read_csv_data_(data_path)
    X_train, X_test, y_train, y_test = split_train_test(data, label)

    # logistic regression
    predict_label = logistic_regression_(X_train, y_train, X_test)
    print("logistic regression")
    compute_result_(y_test, predict_label)
    print("\n")

    # decision tree
    predict_label = naive_bayes_(X_train, y_train, X_test)
    print("decision tree")
    compute_result_(y_test, predict_label)
    print("\n")

    # random forest
    predict_label = random_forest_(X_train, y_train, X_test)
    print("random forest")
    compute_result_(y_test, predict_label)
    print("\n")

    # ada boost
    predict_label = ada_boost_(X_train, y_train, X_test)
    print("ada boost")
    compute_result_(y_test, predict_label)
    print("\n")

    # naive bayes
    data = classify_continuous_to_category(data)
    X_train, X_test, y_train, y_test = split_train_test(data, label)
    predict_label = naive_bayes_(X_train, y_train, X_test)
    print("naive bayes")
    compute_result_(y_test, predict_label)


if __name__ == '__main__':

    main()
