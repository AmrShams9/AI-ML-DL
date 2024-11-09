import numpy as np

ofInterest = np.array([
                5.76260089714,
                7.87666520017,
                9.53163269149,
                9.72801578613,
                5.20002737716,
                0.50133290228,
                8.58820041647,
                9.65056792475,
                3.07043110493,
                1.13232332178
              ])
print(ofInterest)

fits = np.random.random((100,10)) * 10

def sortByFitness():
    global fits
    fitness = np.sum((fits - ofInterest)**2,axis=1)
    fits = fits[fitness.argsort()]

def cross(rate):
    for i in range(10,100):
        parents = fits[np.random.random_integers(0,9,2)]
        if (i*np.random.random()) > rate:
            crossPoint = np.random.random_integers(0,10)
            fits[i] = np.hstack((parents[0,:crossPoint],parents[1,crossPoint:]))
        else:
            fits[i] = parents[0]

def mutate(rate):
    for i in range(10,100):
        for gene in range(10):
            if np.random.random() < rate:
                fits[i][gene] = np.random.random()*10

for i in range(100):
    sortByFitness()
    cross(.6)
    mutate(.4)

print(fits[0])