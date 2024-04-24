import numpy as np

FILE_PATH = "./data/g4.in"


def r0(graph: np.ndarray) -> int:
    """
    If the input graph is empty, return 0.
    If the input graph G has a vertex v without any neighbors, return 1 + R0(G[V - v]).
    Otherwise find a vertex u of maximum degree and return max(1 + R0(G[V - N[u]]), R0(G[V - u]))
    """
    print("-"*10)
    print(graph)
    if np.all(graph == -1):
        return 0

    # Find row where all elements are equal to 0
    v = np.where(np.all(graph == 0, axis=1))[0]
    print("v = ", v)
    if v.size > 0:
        v = v[0].item()

        # new_graph = np.delete(np.delete(graph, v, axis=0), v, axis=1)
        GV_v = graph.copy()
        GV_v[v] = -1
        GV_v[...,v] = -1
        return 1 + r0(GV_v)

    # Find vertex with maximum degree
    u = np.argmax(np.sum(graph))
    neighbours = np.append(np.where(graph[u] == 1), u)
    print("u = ", u)
    print("neighbours = ", neighbours)

    GV_Nu = graph.copy()
    GV_Nu[neighbours] = -1
    GV_Nu[...,neighbours] = -1

    GV_u = graph.copy()
    GV_u[u] = -1
    GV_u[...,u] = -1
    return max(1 + r0(GV_Nu), r0(GV_u))


if __name__ == "__main__":
    with open(FILE_PATH) as f:
        n = int(f.readline())  # Number of vertices
        data = f.readlines()  # Adjacency matrix

    adjacency_matrix = np.array([[int(x) for x in line.split()] for line in data])
    result = r0(adjacency_matrix)
    print(result)
