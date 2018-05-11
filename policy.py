import numpy as np
import node


class MCTS:

    label = 'MCTS'

    def __init__(self, model, horizon_remain=0, budget=10, alpha=2):
        self.budget = budget
        # alpha is a tunable parameter in the UCT algorithm
        self.alpha = alpha
        self.node = node.Node(model, horizon_remain)

    # update the state information
    def update(self, model_new, horizon_remain):
        self.node = node.Node(model_new, horizon_remain)

    # tree_policy descends the tree to a leaf node of the tree
    def tree_policy(self):
        # descend the tree if the remaining horizon is positive and the current node is not a leaf node
        while self.node.horizon_remain > 0 and (len(self.node.children_node) > 0 or self.node.parent_node is None):
            if len(self.node.untried_decision) > 0:
                self.expand()
            else:
                self.node = self.best_child(self.alpha)

    # adds a child node and sets the current node to the child
    def expand(self):
        model_temp = self.node.model.__copy__()
        # randomly choose an untried decision
        index = int(np.random.rand() * len(self.node.untried_decision))
        decision = self.node.untried_decision[index]
        # remove the decision from the set of untried decisions
        self.node.untried_decision.pop(index)
        link_reward = model_temp.reward(decision)
        model_temp.forward_one_step(decision)
        child_node = node.Node(model_temp, self.node.horizon_remain-1, self.node, decision, link_reward)
        self.node.add_child(child_node)
        self.node = child_node

    # returns the best child node using the UCT algorithm
    def best_child(self, c):
        max_score = None
        best_child = None
        for child_node in self.node.children_node:
            # compute the UCT score
            score = child_node.link_reward + child_node.value/child_node.visits\
                    + c*np.sqrt(2*np.log(self.node.visits)/child_node.visits)
            if max_score is None or score > max_score:
                max_score = float(score)
                best_child = child_node
        return best_child

    # simulation() takes a node of the tree and simulates a path toward a terminal state using the default policy
    # simulation() returns the cumulative reward collected in one sample path
    def simulate(self):
        model_temp = self.node.model.__copy__()
        reward = 0
        for _ in range(self.node.horizon_remain):
            decision = self.default_policy(model_temp)
            reward += model_temp.reward(decision)
            model_temp.forward_one_step(decision)
        return reward

    # the default policy selects an action at random from the feasible set
    def default_policy(self, model_temp=None):
        if model_temp is None:
            model_temp = self.node.model.__copy__()
        index = int(np.random.rand() * len(model_temp.decision_range))
        return model_temp.decision_range[index]

    # backup the value for all parent nodes until the root node is reached
    def backup(self, reward):
        while True:
            self.node.visits += 1
            self.node.value += reward
            if self.node.parent_node is not None:
                self.node = self.node.parent_node
            else:
                break

    # the decision function returns a decision using the MCTS algorithm
    def decision(self):
        # return a random decision if the budget is 0
        if self.budget == 0:
            return self.default_policy(self.node.model)
        for _ in range(self.budget):
            self.tree_policy()
            reward = self.simulate()
            self.backup(reward)
        return self.best_child(0).link_decision


class PureExploration:

    label = 'Pure Exploration'

    def __init__(self, model):
        self.model = model

    def update(self, model_new, _):
        self.model = model_new

    def decision(self):
        index = int(np.random.rand()*len(self.model.decision_range))
        return self.model.decision_range[index]
