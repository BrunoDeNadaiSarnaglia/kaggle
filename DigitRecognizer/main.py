__author__ = 'Bruno'
import csv
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

def main():
    f = open("train.csv", "rb")
    reader = csv.reader(f)
    y = []
    x = []
    for row in reader:
        y.append(row[0])
        x.append(row[1:])
    clf = neighbors.KNeighborsClassifier()
    clf2 = GridSearchCV(estimator=clf, param_grid=dict(), cv=5)

    clf2.fit(x[1:], y[1:])
    print clf2.best_score_

    f = open("test.csv", "rb")
    reader = csv.reader(f)
    x = []
    for row in reader:
        x.append(row)
    y = clf2.best_estimator_.predict(x[1:])

    f = open("output.csv", "wb")
    f.write("ImageId,Label\n")
    i = 1
    for p in y:
        f.write(str(i) + "," + str(p) + "\n")
        i += 1
    f.close()

main()