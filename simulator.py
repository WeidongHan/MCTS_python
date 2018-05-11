import numpy as np
import matplotlib.pyplot as plt
import csv


class Simulator:

    # initialize the simulator, setting default values of time horizon and number of paths to 10
    # reward stores the marginal reward in all sample paths
    # cum_reward stores the cumulative reward (starting with 0) in all sample paths
    def __init__(self, model, policy, horizon=10, num_path=10):
        self.horizon = horizon
        self.num_path = num_path
        self.model = model
        self.policy = policy
        self.decision = []
        self.reward = []
        self.cum_reward = []

    # this function generates a single sample path
    def one_path_gen(self):
        # define model_temp to store the temporary state information
        model_temp = self.model.__copy__()
        reward_one_path = [0 for _ in range(self.horizon)]
        decision_one_path = [0 for _ in range(self.horizon)]
        for i in range(self.horizon):
            # update the state information used by the policy
            self.policy.update(model_temp, self.horizon-i)
            decision_one_path[i] = self.policy.decision()
            reward_one_path[i] = model_temp.reward(decision_one_path[i])
            model_temp.forward_one_step(decision_one_path[i])
        return reward_one_path, decision_one_path

    # generate all sample paths
    def all_path_gen(self):
        for _ in range(self.num_path):
            reward_one_path, decision_one_path = self.one_path_gen()
            cum_reward_one_path = np.insert(np.cumsum(reward_one_path), 0, 0)
            self.reward.append(reward_one_path)
            self.decision.append(decision_one_path)
            self.cum_reward.append(cum_reward_one_path)

    # saving the results in a .csv file
    def output_csv(self):
        filename = 'data/output_' + self.model.label + '_' + self.policy.label + '.csv'
        with open(filename, 'w', newline='') as csv_out:
            wf = csv.writer(csv_out)
            for n in range(self.num_path):
                header = 'Path_' + str(n+1)
                wf.writerow([header+'_decision'] + [str(j) for j in self.decision[n]])
                wf.writerow([header+'_reward'] + [str(j) for j in self.reward[n]])

    # plot the average cumulative reward
    def avg_reward_plot(self):
        if len(self.cum_reward) == 0:
            return
        avg_reward = np.mean(self.cum_reward, axis=0)
        plt.plot(list(range(self.horizon+1)), avg_reward, 'b-', lineWidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Average cumulative reward')
        filename = 'plots/Average_reward_' + self.model.label + '_' + self.policy.label + '.pdf'
        plt.savefig(filename, format='pdf')
        plt.clf()
