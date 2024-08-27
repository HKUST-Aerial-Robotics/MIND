class Node:
    def __init__(self, key, parent_key, data):
        self.key = key
        self.parent_key = parent_key
        self.children_keys = []
        self.data = data
        self.depth = 0

    def __str__(self):
        return f"Node_{self.key}: Parent: {self.parent_key}, Children: {self.children_keys}"


class Tree:
    def __init__(self):
        self.nodes = {}
        self.root = None
        self.leaves = []

    def get_node(self, key):
        if key in self.nodes:
            return self.nodes[key]
        else:
            raise KeyError("Node with the given key does not exist.")

    def get_parent_key(self, key):
        if key in self.nodes:
            return self.nodes[self.nodes[key].parent_key]
        else:
            raise KeyError("Node with the given key does not exist.")

    def has_children(self, key):
        if key in self.nodes:
            return len(self.nodes[key].children_keys) > 0
        else:
            raise KeyError("Node with the given key does not exist.")

    def get_children_keys(self, key):
        if key in self.nodes:
            return self.nodes[key].children_keys
        else:
            raise KeyError("Node with the given key does not exist.")

    def get_root(self):
        if self.root is not None:
            return self.nodes[self.root]
        else:
            raise KeyError("root node does not exist.")

    def get_root_key(self):
        if self.root is not None:
            return self.root
        else:
            raise KeyError("root node does not exist.")

    def add_node(self, node):
        if node.parent_key is None and self.nodes == {}:
            self.nodes[node.key] = node
            self.root = node.key
            self.leaves.append(node.key)
        else:
            if node.parent_key not in self.nodes:
                raise KeyError("Parent does not exist.")
            if node.key in self.nodes:
                raise ValueError("Node key already exists.")
            self.nodes[node.parent_key].children_keys.append(node.key)
            if node.parent_key in self.leaves:
                self.leaves.remove(node.parent_key)
            self.nodes[node.key] = node
            self.nodes[node.key].depth = self.nodes[node.parent_key].depth + 1
            self.leaves.append(node.key)

    def print(self):
        self._process_up_down(self.root, print)

    def process_up_down(self, fcn):
        if self.root is None:
            raise KeyError("root node does not exist.")
        self._process_up_down(self.root, fcn)

    def get_leaf_nodes(self):
        leaf_nodes = []
        for l in self.leaves:
            leaf_nodes.append(self.nodes[l])
        return leaf_nodes

    def get_leaf_keys(self):
        return self.leaves

    def retrieve_nodes_to_root(self, key):
        ret_nodes = []
        cur_node = self.get_node(key)
        ret_nodes.append(cur_node)
        while cur_node.parent_key is not None:
            cur_node = self.get_node(cur_node.parent_key)
            ret_nodes.append(cur_node)
        return ret_nodes

    def size(self):
        return len(self.nodes)

    def _process_bottom_up(self, key, fcn):
        current_node = self.nodes[key]
        fcn(current_node)
        self._process_up_down(current_node.parent_key, fcn)

    def _process_up_down(self, key, fcn):
        current_node = self.nodes[key]
        fcn(current_node)
        for child_key in current_node.children_keys:
            self._process_up_down(child_key, fcn)
