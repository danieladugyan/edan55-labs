from time import sleep

import numpy as np

FILE_PATH = "./data/g60.in"
DEBUG = False


class Graph:
    def __init__(self, n: int, g: np.ndarray):
        self.n = n
        self.g = g

    def __add__(self, other):
        g = np.append(np.append(self.g, other, axis=0), other, axis=1)
        return Graph(self.n, g)

    def __sub__(self, other):
        g = np.delete(np.delete(self.g, other, axis=0), other, axis=1)
        return Graph(self.n, g)

    def is_empty(self):
        return self.g.size == 0

    def _find_vertex_with_n_neighbours(self, n):
        v = np.where(np.sum(self.g, axis=0) == n)[0]
        if v.size > 0:
            return v[0].item()

    def neighbours(self, v):
        return np.append(np.where(self.g[v] == 1)[0], v)

    def find_vertex_without_neighbours(self):
        return self._find_vertex_with_n_neighbours(0)

    def find_vertex_with_one_neighbour(self):
        return self._find_vertex_with_n_neighbours(1)

    def find_vertex_with_two_neighbours(self):
        return self._find_vertex_with_n_neighbours(2)

    def find_vertex_with_max_degree(self):
        return np.argmax(np.sum(self.g, axis=0))

    def copy(self):
        return Graph(self.n, self.g.copy())

    def __repr__(self) -> str:
        return self.g.__repr__()


def r(graph: Graph) -> int:
    """
    If the input graph is empty, return 0.
    If the input graph G has a vertex v without any neighbors, return 1 + R0(G[V - v]).
    Otherwise find a vertex u of maximum degree and return max(1 + R0(G[V - N[u]]), R0(G[V - u]))
    """
    if DEBUG:
        print("-" * 20)
        print(graph)
        sleep(1)

    if graph.is_empty():
        if DEBUG:
            print("Empty graph")
        return 0

    v = graph.find_vertex_without_neighbours()
    if DEBUG:
        print("v = ", v)
    
    if v is not None:
        return 1 + r(graph - v)

    # Find vertex with maximum degree
    u = graph.find_vertex_with_max_degree()
    if DEBUG:
        print("u = ", u)
    
    return max(1 + r(graph - graph.neighbours(u)), r(graph - u))


if __name__ == "__main__":
    with open(FILE_PATH) as f:
        n = int(f.readline())  # Number of vertices
        data = f.readlines()  # Adjacency matrix

    adjacency_matrix = np.array([[int(x) for x in line.split()] for line in data])
    graph = Graph(n, adjacency_matrix)

    result = r(graph)
    print(result)
