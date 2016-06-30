def classify(features_train, labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_leaf=4)
    clf.fit(features_train, labels_train)
   
    return clf