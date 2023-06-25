from globals import *


def distance_nodes(from_node, to_node):
    np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)

