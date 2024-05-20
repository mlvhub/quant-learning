# ## Breadth-First Search
#
# A search algorithm that runs on graphs, it can help answer two types of questions:
# - is there a path from node A to B?
# - what is the shortest path from node A to B?

graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []

# +
from collections import deque

# O(V+E) running time (V for number of vertices, E for number of edges).
def shortest_path(node_1, node_2):
    checked = {}
    search_queue = deque([(n, [n]) for n in graph[node_1]])
    while search_queue:
        (n, path) = search_queue.popleft()
        if n == node_2:
            return path
        else:
            checked[n] = True
            search_queue += [(n, path + [n]) for n in graph[n] if n not in checked]
    return None



# -

shortest_path("you", "peggy")

shortest_path("you", "jonny")

shortest_path("anuj", "peggy")


