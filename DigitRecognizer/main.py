__author__ = 'Bruno'
import csv
from sklearn import neighbors


def main():
    f = open("train.csv", "rb")
    reader = csv.reader(f)
    y = []
    x = []
    for row in reader:
        y.append(row[0])
        x.append(row[1:])
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x[1:], y[1:])

main()