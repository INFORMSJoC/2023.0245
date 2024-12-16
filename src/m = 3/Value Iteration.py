import numpy as np
import pandas as pd
from scipy.special import gamma, factorial
import ast
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import multiprocessing as mp
import math
import pickle

# Define the InventoryMDP class to handle inventory-related Markov Decision Processes
class InventoryMDP(object):
    def startState(self):
        # Initial state of the inventory [inventory with remaining age of two days, inventory with remaining age of one day]
        return [0, 0]
    def orders(self, state):
        # Generate a list of valid orders based on the state
        TopUp = list(range(21))
        result = [max(x - sum(state), 0) for x in TopUp]
        return list(set(result))
    def succProbCost(self, state, order, demand, size, prob, day, cs, Plogit):
        # Compute next state, probability, and cost for given inputs
        c02, c03, c12, c13 = Plogit
        # Calculate probabilities using a multinomial logistic regression
        P2 = np.exp(c02 + c12 * order)/(1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
        P3 = np.exp(c03 + c13 * order)/(1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
        P1 = 1 - P2 - P3
        result = []
        # Consider all possible combinations of demand and shelf-life uncertainty
        for i in range(len(demand)):
            for y3 in range(order+1):
                for y2 in range(order-y3+1):
                    y1 = order - y3 - y2
                    # Compute new state
                    newState = [0, 0]
                    newState[1] = max(state[0] + y2 - max(demand[i] - state[1] - y1, 0), 0)
                    newState[0] = max(y3 - max(demand[i] - state[0] - y2 - state[1] - y1, 0), 0)
                    # Compute immediate cost
                    cost = cs[0]*int(order>0) + cs[1]*max(order + state[0] + state[1] - demand[i], 0) + cs[2]*max(demand[i] - order - state[0] - state[1], 0) + cs[3]*max(state[1] + y1 - demand[i], 0)
                    # Compute shelf-life probability using a mutinomial distribution
                    aprob = math.factorial(order)*pow(P1,y1)*pow(P2,y2)*pow(P3,y3)/(math.factorial(y3)*math.factorial(y2)*math.factorial(y1))
                    # Compute demand probability
                    if (demand[i] == 20):
                        dprob = 1-sum(gamma(i+size[day])*(prob[day]**size[day])*(1-prob[day])**i/(gamma(size[day])*factorial(i)) for i in range(20))
                        result.append((newState, dprob, aprob, cost))
                    else:
                        dprob = gamma(demand[i]+size[day])*(prob[day]**size[day])*(1-prob[day])**demand[i]/(gamma(size[day])*factorial(demand[i]))
                        result.append((newState, dprob, aprob, cost))
        return result
    def discount(self):
        # Define discount factor for future costs
        return 0.95
    def states(self):
        # Generate all possible states
        return [[i, j] for i in range(21) for j in range(21-i)]

# Perform value iteration to find optimal value function and policy
def ValueIteration(mdp, demand, size, prob, cs, Plogit):
    # Initialize value function for 7 days
    V = [{} for _ in range(7)]
    for i in range(7):
        for state in mdp.states():
            V[i][str(state)] = 0.0

    def Q(state, order, demand, day):
        # Compute exact expected cost for a given state and order
        return sum(dprob * aprob * (cost + mdp.discount() * V[(day + 1) % 7][str(newState)])
                   for newState, dprob, aprob, cost in mdp.succProbCost(state, order, demand, size, prob, day, cs, Plogit))

    day = 0
    # Initialize new values
    newV = [{} for _ in range(7)]
    for i in range(7):
        for state in mdp.states():
            newV[i][str(state)] = 0.0

    while True:
        # Update value function
        for state in mdp.states():
            newV[day][str(state)] = min(Q(state, order, demand, day) for order in mdp.orders(state))

        # Check for convergence
        if (day == 6):
            if max(abs(V[day][str(state)] - newV[day][str(state)]) for day in range(7) for state in mdp.states()) < 0.1:
                break
            else:
                V = newV
                day = 0
                # Initialize new values
                newV = [{} for i in range(7)]
                for i in range(7):
                    for state in mdp.states():
                        newV[i][str(state)] = 0.0
        else:
            day += 1

    # Extract policy
    pi = [{} for _ in range(7)]
    for day in range (7):
        for state in mdp.states():
            pi[day][str(state)] = min((Q(state, order, demand, day), order) for order in mdp.orders(state))[1]

    return V, pi

# Implementation function to calculate value function and policy
def implementation(cs, Plogit):
    # Define negative-binomial demand parameters
    size = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
    prob = [size[0] / (size[0] + 5.660569), size[1] / (size[1] + 6.922555), size[2] / (size[2] + 6.504332),
            size[3] / (size[3] + 6.165049), size[4] / (size[4] + 5.816060), size[5] / (size[5] + 3.326408),
            size[6] / (size[6] + 3.426814)]
    d = list(range(21))   # Demand range

    # Perform value iteration
    mdp = InventoryMDP()
    OVFDic, OPDic = ValueIteration(mdp, demand=d, size = size, prob = prob, cs = cs, Plogit = Plogit)
    OVF = pd.DataFrame.from_dict(OVFDic, orient='columns')
    OP = pd.DataFrame.from_dict(OPDic, orient='columns')

    return {'Cost': str(cs), 'Endogeneity': str(Plogit), 'OVF':OVF, 'OP': OP}

# Input parameters for cost and multinomial logit coefficients
c = [[10, 1, 20, 5], [10, 1, 20, 20], [10, 1, 20, 80], [100, 1, 20, 5], [100, 1, 20, 20],[100, 1, 20, 80]]
c01 = [[1, 0.5, -0.4, -0.8], [1, 0.5, -0.2, -0.1], [1, 0.5, -0.1, -0.05], [1, 0.5, 0.0, 0.0], [1, 0.5, 0.2, 0.4], [1, 0.5, 0.4, 0.8]]

if __name__ ==  '__main__':
    # Parallel processing for implementation
    pool = mp.Pool(processes=40)
    results = [pool.apply_async(implementation, args=(x, y)) for x in c for y in c01]
    output = [p.get() for p in results]
    # Save results to a pickle file
    open_file = open("OP_Paper_Final.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()