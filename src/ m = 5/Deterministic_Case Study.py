import numpy as np
from scipy.special import gamma, factorial
import multiprocessing as mp
import math
import pickle

# Define the InventoryMDP class to handle inventory-related Markov Decision Processes
class InventoryMDP(object):
    def startState(self):
        # Initial state of the inventory [inventory with remaining age of four days, ..., inventory with remaining age of one day]
        return [0, 0, 0, 0]
    def orders(self, state):
        # Generate a list of valid orders based on the state
        TopUp = list(range(21))
        result = [max(x - sum(state), 0) for x in TopUp]
        return list(set(result))
    def succProbCost(self, state, order, demand, day, cs):
        # Compute next state, probability, and cost for given inputs
        result = []
        # Consider all possible combinations of demand with the deterministic maximum shelf-life of five days
        for i in range(len(demand)):
            # Compute new state
            newState = [0, 0, 0, 0]
            newState[3] = max(state[2] - max(demand[i] - state[3], 0), 0)
            newState[2] = max(state[1] - max(demand[i] - state[3] - state[2], 0), 0)
            newState[1] = max(state[0] - max(demand[i] - state[3] - state[2] - state[1], 0), 0)
            newState[0] = max(order - max(demand[i] - state[3] - state[2] - state[1] - state[0], 0), 0)
            # Compute immediate cost
            cost = cs[0]*int(order>0) + cs[1]*max(order + sum(state) - demand[i], 0) + cs[2]*max(demand[i] - order - sum(state), 0) + cs[3]*max(state[3] - demand[i], 0)
            # Compute demand probability
            if (demand[i] == 20):
                dprob = 1-sum(gamma(i+size[day])*(prob[day]**size[day])*(1-prob[day])**i/(gamma(size[day])*factorial(i)) for i in range(20))
                result.append((newState, dprob, cost))
            else:
                dprob = gamma(demand[i]+size[day])*(prob[day]**size[day])*(1-prob[day])**demand[i]/(gamma(size[day])*factorial(demand[i]))
                result.append((newState, dprob, cost))
        return result
    def discount(self):
        # Define discount factor for future costs
        return 0.95
    def states(self):
        # Generate all possible states
        return [[i, j, k, l] for i in range(21) for j in range(21-i) for k in range(21-i-j) for l in range(21-i-j-k)]


# Perform value iteration to find optimal value function and policy
def valueIteration(cs):
    # Initialize value function for 7 days
    V = [{} for _ in range(7)]
    for i in range(7):
        for state in mdp.states():
            V[i][str(state)] = 0.0

    def Q(state, order, day, demand= d):
        # Compute exact expected cost for a given state and order
        return sum(dprob * (cost + mdp.discount() * V[(day + 1) % 7][str(newState)])
                   for newState, dprob, cost in
                   mdp.succProbCost(state, order, demand, day, cs))

    day = 0
    # Initialize new values
    newV = [{} for _ in range(7)]
    for i in range(7):
        for state in mdp.states():
            newV[i][str(state)] = 0.0

    while True:
        # Update value function
        for state in mdp.states():
            newV[day][str(state)] = min(Q(state, order, day) for order in mdp.orders(state))

        # Check for convergence
        if (day == 6):
            if max(abs(V[day][str(state)] - newV[day][str(state)]) for day in range(7) for state in mdp.states()) < 0.1:
                break
            else:
                V = newV
                day = 0
                # Initialize new values
                newV = [{} for _ in range(7)]
                for i in range(7):
                    for state in mdp.states():
                        newV[i][str(state)] = 0.0
        else:
            day += 1

    # Extract policy
    pi = [{} for i in range(7)]
    for day in range(7):
        for state in mdp.states():
            pi[day][str(state)] = min((Q(state, order, day), order) for order in mdp.orders(state))[1]

    return V, pi

# Implementation function to calculate value function and policy
def implementation(cs):
    # Perform value iteration
    OV, OP = valueIteration(cs=cs)

    return {'CS': str(cs), 'OVF':OV, 'OP':OP}

# Define maximum shelf-life parameter
m = 5
# Define negative-binomial demand parameters
size = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
prob = [size[0] / (size[0] + 5.660569), size[1] / (size[1] + 6.922555), size[2] / (size[2] + 6.504332),
        size[3] / (size[3] + 6.165049), size[4] / (size[4] + 5.816060), size[5] / (size[5] + 3.326408),
        size[6] / (size[6] + 3.426814)]
d = list(range(21))

mdp = InventoryMDP()

# Input parameters for cost
c = [[0, 1, 20, 2], [0, 1, 20, 5], [0, 1, 20, 20], [10, 1, 20, 2], [10, 1, 20, 5], [10, 1, 20, 20], [20, 1, 20, 2], [20, 1, 20, 5], [20, 1, 20, 20]]

if __name__ ==  '__main__':
    # Parallel processing for implementation
    pool = mp.Pool(processes=40)
    results = [pool.apply_async(implementation, args=(x,)) for x in c]
    output = [p.get() for p in results]
    # Save results to a pickle file
    open_file = open("DetPolicy_m5_CS.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()