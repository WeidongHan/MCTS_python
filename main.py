import model
import policy
import simulator


if __name__ == '__main__':
    p_0 = 5
    r_0 = 30
    sigma_W = 0.2
    horizon = 4
    num_path = 30
    md = model.Model(r_0, p_0, sigma_W)
    # set up budget constraint for MCTS
    budget = 3000
    # tunable parameter in the UCT algorithm
    alpha = 2
    # the following line uses Pure Exploration policy
    # pl = policy.PureExploration(md)
    # the following line uses MCTS policy
    pl = policy.MCTS(md, horizon, budget, alpha)
    sim = simulator.Simulator(md, pl, horizon, num_path)
    sim.all_path_gen()
    sim.avg_reward_plot()
    sim.output_csv()
