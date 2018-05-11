import numpy as np


# define our default model using a selling problem
class Model:

    # name of the model
    label = 'selling'

    def __init__(self, r_0, p_0, sigma_w):
        # state variable includes the current price and inventory level
        self.state = [r_0, p_0]
        self.sigma_w = sigma_w
        self.exo_info = 0
        self.decision_range = list(range(r_0+1))

    # define a copy constructor
    def __copy__(self):
        return Model(self.state[0], self.state[1], self.sigma_w)

    # generate the daily return in price assuming the price follows a log-normal distribution
    def exogenous_gen(self):
        self.exo_info = self.sigma_w*np.random.randn()

    # transition function updates the state variables using the decision x_t and the exogenous information
    def transition(self, x_t):
        if self.state[0] > x_t:
            self.state[0] = self.state[0] - x_t
        else:
            self.state[0] = 0
        self.state[1] *= np.exp(self.exo_info)
        self.decision_range = list(range(self.state[0]+1))

    # move one step forward by generating the exogenous information and updating the state
    def forward_one_step(self, x_t):
        self.exogenous_gen()
        self.transition(x_t)

    # the reward is given by the product of price and the amount we sell
    def reward(self, x_t):
        return self.state[1]*x_t
