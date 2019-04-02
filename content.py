# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    N1, N2 = X.shape[0], X_train.shape[0]
    D1, D2 = X.shape[1], X_train.shape[1]
    dist = np.empty([N1, N2], int)
    for n1 in range(N1):
        for n2 in range(N2):
            dist[n1][n2] = D1 - np.sum(X[n1] == X_train[n2]) #numpy.sum(arr1 == arr2) sumje na ilu pozycjach są takie same elementy
    return dist


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    N1, N2 = Dist.shape[0], Dist.shape[1]
    y_matrix = np.empty([N1, N2], int)  # result matrix
    for n1 in range(N1):
        dist_and_num = zip(Dist[n1], range(N2))  # list of pairs (Dist[n1], n2)
        dist_and_num_sorted = sorted(dist_and_num, key=lambda pair: pair[0])  # sorting based of value of Dist[n1]
        # add the value of y[n2] for each (Dist[n1], n2) pair in sorted array
        y_matrix[n1] = [y[n2] for _, n2 in dist_and_num_sorted]

    return y_matrix


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    N1, N2 = y.shape[0], y.shape[1]
    labels = sorted((set(y[0])))  # set of labels
    prob_matrix = np.empty([N1, len(labels)], float)  # result matrix

    for n1 in range(N1):
        labels_occurs = dict(zip(labels, np.zeros(len(labels))))  # dictionary - label: occurs_no
        for i in range(k):  # counting occurs of kNN
            occurs = labels_occurs[y[n1][i]]
            occurs += 1
            labels_occurs.update({y[n1][i]: occurs})

        row_sum = sum(labels_occurs.values())
        for label in labels_occurs.keys():  # count occurs frequence
            occurs = labels_occurs[label]
            mean = occurs / row_sum
            labels_occurs.update({label: mean})

        prob_matrix[n1] = list(labels_occurs.values())  # convert dict to list

    return prob_matrix


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    N, M = p_y_x.shape[0], p_y_x.shape[1]
    y_predict = [M-np.argmax(np.flip(p))-1 for p in p_y_x]
    err = sum(y_true != y_predict)/N
    return err

def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    dist = hamming_distance(X_val, X_train)
    y_matrix = sort_train_labels_knn(dist, y_train)
    ks_and_errors = [(k, classification_error(p_y_x_knn(y_matrix, k), y_val)) for k in k_values]
    best_k, best_error = min(ks_and_errors, key=lambda k_and_err: k_and_err[1])
    _, errors = zip(*ks_and_errors)
    return best_error, best_k, errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    labels = set(y_train)
    labels_and_prob = dict(zip(labels, np.zeros(len(labels))))
    for y in y_train:
        if y in labels:
            ys_number = labels_and_prob[y]
            labels_and_prob.update({y: (ys_number+1)})

    for label in labels:
        prob = labels_and_prob[label]/len(y_train)
        labels_and_prob.update({label: prob})

    return list(labels_and_prob.values())


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    pass


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.

    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    pass
