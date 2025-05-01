from graphviz import Digraph
from isolation_forest import IsolationForest


def visualize_iTree(tree, dot=None, parent_id=None, node_id=0):
    if dot is None:
        dot = Digraph(comment="Isolation Tree")
    # label the node with its split criterion or leaf size
    if tree.get("is_leaf", False):
        label = f"Isolated: {tree['item_value']} \n depth={tree['depth']}"
        fillcolor = "gray"
    else:
        i = tree["input_column_index"]
        sv = tree["split_value"]
        label = f"x[{i}] < {sv:.2f}"
        fillcolor = "lightblue"

    this_id = str(node_id)
    dot.node(this_id, label, style="filled", fillcolor=fillcolor)

    if parent_id is not None:
        dot.edge(parent_id, this_id)

    if not tree.get("is_leaf", False):
        # left subtree
        node_id = visualize_iTree(tree["left"], dot, this_id, int(this_id) * 2 + 1)
        # right subtree
        node_id = visualize_iTree(tree["right"], dot, this_id, int(this_id) * 2 + 2)
    return node_id + 1


x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, -1, 43]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
input_columns = [x0]  # only features
solution_column = y  # labels (not used in training)

# Create and fit the model
iforest = IsolationForest(num_trees=100, subsample_size=256, random_seed=0)
iforest.fit(input_columns)
one_tree = iforest.forest[0]
dot = Digraph(format="png")
visualize_iTree(one_tree, dot)

x0_values_text = "x[0] values:\n[" + ", ".join(str(v) for v in x0) + "]"
dot.node(
    "legend", x0_values_text, shape="note", style="filled", fillcolor="lightyellow"
)


dot.render("iforest_tree_0", view=True)
