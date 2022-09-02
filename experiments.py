from agents import RandomAgent
from arena import run_arena

# Experiment #1 playing randomly does player 1 have any advantage
# -----Result {0: 538058, 1: 443555, 2: 18387}
# player0, player1, tie respectively
num_trials = 1000000
# outcomes = {0:0, 1:0, 2:0}
# for i in range(num_trials):
# random_agent = RandomAgent()
# winner = run_arena(random_agent, random_agent, render=False)
# outcomes[winner] += 1

# print(outcomes)

# outcomes = {0: 538058, 1: 443555, 2: 18387}
# for i in range(3):
# print(f"Percent {i}:",outcomes[i]/num_trials)
# Percent 0: 0.538058
# Percent 1: 0.443555
# Percent 2: 0.018387

# Experiment #2
