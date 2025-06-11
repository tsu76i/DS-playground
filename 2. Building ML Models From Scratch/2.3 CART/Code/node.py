class Node:
    """
    A class representing a node in the decision tree.

    Attributes:
        type (str): Type of the node ("leaf" or "node").
        value (int or None): Predicted label for leaf nodes. None for internal nodes.
        feature (int or None): Index of the feature for internal nodes. None for leaf nodes.
        threshold (float or None): Threshold value for internal nodes. None for leaf nodes.
        left (Node or None): Left child node. None for leaf nodes.
        right (Node or None): Right child node. None for leaf nodes.
    """

    def __init__(self, type: str, value: float = None,
                 feature: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None):
        """
        Initialise a Node instance.

        Args:
        type (str): Type of the node ("leaf" or "node").
        value (int or None): Predicted label for leaf nodes. None for internal nodes.
        feature (int or None): Index of the feature for internal nodes. None for leaf nodes.
        threshold (float or None): Threshold value for internal nodes. None for leaf nodes.
        left (Node or None): Left child node. None for leaf nodes.
        right (Node or None): Right child node. None for leaf nodes.
        """
        self.type = type  # "leaf" or "node"
        self.value = value  # For leaf nodes
        self.feature = feature  # For internal nodes
        self.threshold = threshold  # For internal nodes
        self.left = left  # Left child
        self.right = right  # Right child
