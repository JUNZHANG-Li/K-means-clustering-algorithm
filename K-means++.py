import numpy as np
import matplotlib.pyplot as plt

# parameter assignment
k = 4
# converge_time = 20

# loading points from npy file
a = np.load('clustering_data.npy')
data = a['data']

# show original data set
plt.scatter(data[:, 0], data[:, 1])

# randomly pick up one mean
idx = np.random.randint(0, 220, 1)
plt.scatter(data[idx, 0], data[idx, 1])
valuelist = []

k -= 1
mean = data[idx, :]  # the import
record = data[idx, :]

# find rest of means
for i in range(k):
    recmean = mean
    # find mean's distance
    a = data[:, :, np.newaxis]
    mean = mean[:, :, np.newaxis]  # increase the dimension
    mean = np.transpose(mean, (2, 1, 0))  # transpose
    new = a - mean  # broadcast
    dist = np.linalg.norm(new, axis=1)  # normalisation
    dist = dist.min(1)  # pick up the minimum distant

    # pick next mean depend on probability
    list = np.arange(220)  # index list
    sum = np.sum(dist)
    dist = np.true_divide(dist, sum)  # find probability
    next = np.random.choice(a=list, size=1, p=dist)  # random choose by probability

    # concatenate
    next = data[next, :]
    mean = np.concatenate((recmean, next), axis=0)
    record = np.concatenate((record, next), axis=0)

x_classify = []
y_classify = []
previousTotal = np.zeros((k + 1, 3))
loop = 0
converged = False
maxConvergeTime = 100
loopCounter = 0

# converge
# for z in range(converge_time):
while not converged:
    a = data[:, :, np.newaxis]
    mean = mean[:, :, np.newaxis]  # increase the dimension
    mean = np.transpose(mean, (2, 1, 0))  # transpose
    new = a - mean  # broadcast
    dist = np.linalg.norm(new, axis=1)  # normalisation
    dist1 = np.min(dist, axis=1)
    dist = np.argmin(dist, axis=1)  # pick up the minimum distant

    total = np.zeros((k + 1, 3))
    loopCounter += 1
    x_classify = np.zeros((k + 1, 220))  # record x-axis of points in one class
    y_classify = np.zeros((k + 1, 220))  # record y-axis of points in one class

    # defining middle point by x-axis and y-axis of each class of points
    for i in range(220):
        x = 0
        for count in range(k + 1):
            if dist[i] == count:
                while x_classify[count, x] != 0:
                    x = x + 1
                x_classify[count, x] = data[i, 0]
                y_classify[count, x] = data[i, 1]
                total[count, 0] = total[count, 0] + data[i, 0]
                total[count, 1] = total[count, 1] + data[i, 1]
                total[count, 2] = total[count, 2] + 1
    mean = np.zeros((k + 1, 2))
    for i in range(k + 1):
        total[i, 0] = total[i, 0] / total[i, 2]
        total[i, 1] = total[i, 1] / total[i, 2]
        mean[i, 0] = total[i, 0]
        mean[i, 1] = total[i, 1]

    convergeCheck = total == previousTotal
    if not (False in convergeCheck):
        converged = True
        print("Number of loop iterated to reach convergence: ", loopCounter)
    if loopCounter == maxConvergeTime:
        print("maximum converge time is reached, stop converge")
        break
    previousTotal = total

# cancel extra 0 in both x_classify and y_classify
color = ['r', 'g', 'b', 'y', 'violet', 'orange', 'cyan', 'black']
for i in range(k + 1):
    x = x_classify[i, :]
    x = x[x != 0]
    y = y_classify[i, :]
    y = y[y != 0]
    plt.scatter(x, y, c=color[i])

# output image
plt.show()
