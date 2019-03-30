import numpy as np
from collections import Counter

Y = [[1, 2, 3, 1, 2],
     [1, 1, 1, 3, 1],
     [2, 2, 3, 3, 1]]

Z = sorted(set(Y[0]))
print(Z, len(Z))
L = list(zip(Z, np.zeros(len(Z), int)))
D = dict(L)
print(D)
results = np.zeros([len(Y), len(Z)], float)

it = 0
for y in Y:
    D = dict(L)
    for el in y:
        if el in D:
            i: int = D[el]
            i += 1
            D.update({el: i})

    sum_of = sum(D.values())
    print(D)

    for k in D:
        v: int = D[k]
        mean = v/sum_of
        D.update({k: mean})

    results[it] = list(D.values())
    it+=1
    print(D)

print(results)

print(results)
















