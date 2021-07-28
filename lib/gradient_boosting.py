from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import json
import dataset_utils
import datasets

with open('../reports/parameters_random_forest.json', 'r') as file:
    args_dict = json.load(file)

for i in range(len(args_dict)):
    train_set = datasets.Genes('../data', k=args_dict[i]['k'], genes_dict=args_dict[i]['genes_dict'],
                               encoding=dataset_utils.encode_base_occurrence, transpose_output=False)
    train_set, test_set = dataset_utils.split_dataset(train_set, test_size=0.1, shuffle=True)

    x_train = []
    x_test = []
    for entry_train in train_set:
        x_train.append(entry_train.numpy())

    for entry_test in test_set:
        x_test.append(entry_test.numpy())

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np.array(train_set.y)
    y_test = np.array(test_set.y)

    model = GradientBoostingClassifier(n_estimators=args_dict[i]['n_estimators'],
                                       max_depth=args_dict[i]['max_depth'],
                                       random_state=0)

    scores = cross_val_score(model, x_train, y_train, cv=10)
    print(scores)