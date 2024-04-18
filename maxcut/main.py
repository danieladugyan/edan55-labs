import random

NUMBER_OF_RUNS = 100

V: set[int] = set()  # vertices
E: set[tuple[int, int]] = set()  # edges
w: dict[tuple[int, int], int] = {}  # weight of edge


def cut(A: set):
    """
    The cut of a subset A of V is the sum of the weights of the edges
    that have one vertex in A and the other outside of A.
    """
    cut = 0
    for e in E:
        if (e[0] in A and e[1] not in A) or (e[1] in A and e[0] not in A):
            cut += w[e]
    return cut


def r():
    """
    Let A be a random subset of V constructed by flipping a coin
    r(v) ∈ {0, 1} for every vertex v ∈ V and setting v ∈ A if and only if
    r(v) = 1.
    """
    A = set()
    for v in V:
        if random.randint(0, 1) == 1:
            A.add(v)
    return A


def s(A: set[int] = set()):
    """
    Let all the vertices be outside of A to begin with. A vertex
    can be swapped, which means that if it's outside of A its moved
    into A and if its inside A its moved out of A. Pick the first vertex you
    can find that increases your cut if swapped. Swap this vertex, and
    continue doing so until no vertex increases the cut if swapped, eg
    you find a local maxima.
    """
    while True:
        reached_minima = True

        for v in V:
            if cut(A) < cut(A | {v}):
                A.add(v)
                reached_minima = False

            if cut(A) < cut(A - {v}):
                A.remove(v)
                reached_minima = False

        if reached_minima:
            break

    return A


def rs():
    """
    Combine Algorithm R and S by instead of
    placing all vertices outside of A to begin with in S, place the vertices
    according to the output of R. Then proceed with the swapping part
    of S.
    """
    A = r()
    s(A)
    return A


if __name__ == "__main__":
    average_cutsize = 0
    max_cutsize = 0
    with open("./data/matching_1000.txt") as f:
        data = f.readlines()[1:]

    for i in range(NUMBER_OF_RUNS):
        for line in data:
            v1, v2, weight = line.split()
            v1, v2, weight = int(v1), int(v2), int(weight)
            V.add(v1)
            V.add(v2)
            E.add((v1, v2))
            w[(v1, v2)] = weight

        a = cut(rs())
        average_cutsize += a
        max_cutsize = max(max_cutsize, a)

    print(f"Average cutsize: {average_cutsize / NUMBER_OF_RUNS}")
    print(f"Max cutsize: {max_cutsize}")
