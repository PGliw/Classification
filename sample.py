X = [19, 20, 15, 12, 11, 14, 17, 13, 14, 16]
Y = (range(10))
Z = (zip(X, Y))
sorted_by_first = sorted(Z, key=lambda tup: tup[0])
print(sorted_by_first)

list1 = [1, 5, 3]

for i in list1:
    print(i)