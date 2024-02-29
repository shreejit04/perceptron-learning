import sys
import numpy as np
import json


def sgn(z):
    if z < 0.0:
        return -1
    else:
        return 1


def main():
    # read data points
    readfile = open("train-a1-449.txt", "r")
    points = readfile.read()
    readfile.close()

    points = points.split('\n')
    points = points[:-1]
    # print(points)

    # construct numeric data points
    x = []
    y = []
    for index in range(len(points)):
        point = points[index]
        point = point.split(' ')

        y.append(point[-2])
        point = point[:-2]      

        tmp = list(map(float, point))
        tmp = tmp / np.linalg.norm(tmp)  # Multiply vector point by 1/n
        x.append(tmp)

    # Change Y/N to +1/-1
    labels = []
    for label in y:
        if label == 'y':
            labels.append(1)
        else:
            labels.append(-1)
    y = labels
    # print(y)

    # Perceptron learning algorithm
    dim = len(x[0])
    w = np.zeros(dim)
    # print("Normal = ", w)

    flag = False
    count = 0

    weights = []

    while True:
        flag = False
        for index in range(len(x)):
            if y[index] != sgn(np.dot(w, x[index])):
                w = w + np.multiply(y[index], x[index])
                print("Normal = ", w)
                count += 1
                flag = True
        if not flag:
            break

    with open('weight_classifier.txt', 'w') as file:
        file.write(" ".join(map(str, w)))

    print("count = " + str(count))

    min_ = 0
    for point in range(len(x)):
        dis = np.dot(w, x[point])
        if point == 0:
            min_ = dis
        if abs(dis) < abs(min_):
            min_ = dis

    print(min_)


if __name__ == "__main__":
    main()


