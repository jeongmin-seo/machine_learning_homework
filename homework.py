def read_csv_data_(_data_path):

    f = open(_data_path, 'r')
    lines = f.readlines()

    data_type = {
        'category': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 41, 42, 43, 44, 45, 46, 47,
                     48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59],
        'float': [17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'integer': [1, 14, 15, 16, 18, 19, 20, 53],

    }

    data = []
    class_label = []
    for line in lines:
        split_line = line.split(',')

        if 'NA' in split_line:
            continue

        for i, element in enumerate(split_line):
            if i in data_type['category']:
                split_line[i] = int(element)

            elif i in data_type['float']:
                split_line[i] = float(element)

            else:
                split_line[i] = int(element)

        data.append([split_line[0:59]])
        class_label.append(split_line[59])
    f.close()

    return data, class_label


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


def split_train_test(_data, _label):
    from sklearn.model_selection import train_test_split

    return train_test_split(_data, _label, test_sizie=0.33, random_state=72170233)


def compute_average_precision_recall_(_test_label, _predict_label):
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(_test_label, _predict_label)
    recall = recall_score(_test_label, _predict_label)

    print("Precision:", precision, "Recall:", recall)


if __name__ == '__main__':

    data = 'D:\\workspace\\github\\machine_learning_homework\\MLdata2_R.csv'
    data, label = read_csv_data_(data)

    X_train, X_test, y_train, y_test = split_train_test(data, label)
    predict_label = logistic_regression_(X_train, y_train, X_test)
