class Node:

    # initialize an MCTS node
    def __init__(self, model, horizon_remain=0, parent_node=None, link_decision=None, link_reward=None):
        self.model = model.__copy__()
        self.parent_node = parent_node
        self.children_node = []
        self.horizon_remain = horizon_remain
        self.value = 0
        # all the decisions available for this node
        self.decisions = model.decision_range
        # all the untried decisions
        self.untried_decision = model.decision_range
        # the number of times this node has been visited
        self.visits = 0
        # the decision that leads to this node (if the node is not a root node)
        self.link_decision = link_decision
        # the one-period reward collected using the link_decision from the parent node
        self.link_reward = link_reward

    # add a child node to the current node
    def add_child(self, new_child):
        self.children_node.append(new_child)

    # set the value of the node
    def set_value(self, value):
        self.value = value
