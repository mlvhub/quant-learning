# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% [markdown]
# ## Dijktra's Algorithm
#
# Similarly to breadth-first search, Dijkstra's algorith works on graphs but it assigns wegihts to the edges between nodes.
#
# While BFS tells us the shortest path from node A to B, Dijkstra's algorithm tells us the path with the lowest weights. E.g. if the nodes represent places and the edges/weights represent time to travel, Dijktra's will tell us the quickest path from node A to B.

# %%
