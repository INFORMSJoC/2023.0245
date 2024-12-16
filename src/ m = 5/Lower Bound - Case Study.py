import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma, factorial
import ast
import math
import pickle
import multiprocessing as mp

# Multinomial random variable generator
def RV_Multinomial(n, p, U):
    X = np.zeros(len(p), dtype = int)
    for i in range(n):
        for j in range(len(p)):
            if U[i] <= sum(p[k] for k in range(j+1)):
                X[j] = X[j] + 1
                break
    return(X)

# Markov Decision Process (MDP) for inventory control
class InventoryMDP(object):
    def startState(self):
        """
        Define the starting state of the MDP.

        Returns:
            list: Initial state with no inventory ([0, 0, 0, 0]).
        """
        return [0, 0, 0, 0]
    def orders(self, state):
        """
        Determines valid order quantities given the current state.

        Args:
            state (list): Current inventory levels.

        Returns:
            list: Valid order quantities.
        """
        TopUp = list(range(21))
        result = [max(x - sum(state), 0) for x in TopUp]
        return list(set(result))
    def succProbCost(self, state, order, day, cs, sample):
        """
        Computes successor states, their probabilities, and associated costs.

        Args:
            state (list): Current inventory state.
            order (int): Order quantity.
            day (int): Day of the week.
            cs (list): Cost structure parameters.
            sample (list): Random samples for shelf-life realization.

        Returns:
            list: Successor states with probabilities and costs.
        """
        result = []
        for i in range(len(demand)):
            y1, y2, y3, y4, y5 = sample[order]
            newState = [0, 0, 0, 0]
            # Update inventory states based on demand and order
            newState[3] = max(state[2] + y2 - max(demand[i] - state[3] - y1, 0), 0)
            newState[2] = max(state[1] + y3 - max(demand[i] - state[3] - state[2] - y1 - y2, 0), 0)
            newState[1] = max(state[0] + y4 - max(demand[i] - state[3] - state[2] - state[1] - y1 - y2 - y3, 0), 0)
            newState[0] = max(y5 - max(demand[i] - state[3] - state[2] - state[1] - state[0] - y1 - y2 - y3 - y4, 0), 0)
            # Compute cost based on ordering, holding, shortage, and wastage costs
            cost = cs[0]*int(order>0) + cs[1]*max(order + sum(state) - demand[i], 0) + cs[2]*max(demand[i] - order - sum(state), 0) + cs[3]*max(state[3] + y1 - demand[i], 0)
            # Compute demand probabilities
            if (demand[i] == 20):
                dprob = 1-sum(gamma(i+size[day])*(prob[day]**size[day])*(1-prob[day])**i/(gamma(size[day])*factorial(i)) for i in range(20))
                result.append((newState, dprob, cost))
            else:
                dprob = gamma(demand[i]+size[day])*(prob[day]**size[day])*(1-prob[day])**demand[i]/(gamma(size[day])*factorial(demand[i]))
                result.append((newState, dprob, cost))
        return result

    def states(self):
        """
         Generate all possible inventory states.

         Returns:
             list: All feasible inventory states.
         """
        return [[i, j, k, l] for i in range(21) for j in range(21-i) for k in range(21-i-j) for l in range(21-i-j-k)]

# Perform backward induction for value computation
def BackwardInduction(cs, T, sample, dow):
    """
    Solve the MDP using backward induction.

    Args:
        cs (list): Cost structure parameters.
        T (int): Random horizon length.
        sample (list): Random shelf-life samples.

    Returns:
        dict: Lower-bound for all inventory states at day 0.
    """
    V = {str(state): 0.0 for state in mdp.states()}  # Initialize value function

    def Q(state, order, day, t):
        # Compute expected cost value
        return sum (dprob*(cost + V[str(newState)])
                    for newState, dprob, cost in mdp.succProbCost(state, order, day, cs, sample[t]))
    t = T-1
    while t >= 0:
        newV = {}
        day = (t+dow)%7     # Determine day of the week
        for state in mdp.states():
            newV[str(state)] = min(Q(state, order, day, t) for order in mdp.orders(state))
        V = newV
        t -= 1
    return V

def Sample (Plogit, length, Replication):
    """
    Generates sample paths based on a multinomial logit model.

    Args:
        Plogit (list): Logit model parameters.
        length (list): Length of the sample path for each replication.
        Replication (int): Number of replications to simulate.

    Returns:
        list: A list of sampled paths for each replication.
    """
    rng2 = np.random.RandomState(seed=2)
    U = [rng2.random((length[i], 20)) for i in range(Replication)]
    c02, c03, c04, c05, c12, c13, c14, c15 = Plogit
    sample_list =[]
    for rep in range(Replication):
        sample_path = []
        for h in range (length[rep]):
            sample = []
            for i in range(21):
                order = i
                P2 = np.exp(c02 + c12 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order) + np.exp(c04 + c14 * order) + np.exp(c05 + c15 * order))
                P3 = np.exp(c03 + c13 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order) + np.exp(c04 + c14 * order) + np.exp(c05 + c15 * order))
                P4 = np.exp(c04 + c14 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order) + np.exp(c04 + c14 * order) + np.exp(c05 + c15 * order))
                P5 = np.exp(c05 + c15 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order) + np.exp(c04 + c14 * order) + np.exp(c05 + c15 * order))
                P1 = 1 - P2 - P3 - P4 - P5
                # Generate a random sample using a multinomial distribution
                sample.append(RV_Multinomial(i, [P1, P2, P3, P4, P5], U[rep][h]))
            sample_path.append(sample)
        sample_list.append(sample_path)

    return  sample_list

# Main implementation function
def implementation(cs, dayofweek):
    """
    Simulate the perishable inventory problem and compute lower-bound.

    Args:
        cs (list): Cost structure.
        dayofweek (int): Day of the week (0 to 6) for calculating the lower bound.

    Returns:
        list: A list of dictionaries containing the lower-bound values for all inventory states at day = dayofweek.
    """
    rng1 = np.random.RandomState(seed=1)
    # Parameters for the multinomial logit model
    Plogit = [1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]
    Replication = 4000
    length = rng1.geometric(0.05, size=Replication)
    RemAge = Sample(Plogit, length, Replication)
    res_list = []
    for z in range(0, 100):
        LBV = BackwardInduction(cs=cs, T=length[z], sample=RemAge[z], dow=dayofweek)
        Dic = {'LBV': LBV}
        res_list.append(Dic)
    return (res_list)

# Define maximum shelf-life and daily demand distributions
m = 5
size = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
prob = [size[0] / (size[0] + 5.660569), size[1] / (size[1] + 6.922555), size[2] / (size[2] + 6.504332),
        size[3] / (size[3] + 6.165049), size[4] / (size[4] + 5.816060), size[5] / (size[5] + 3.326408),
        size[6] / (size[6] + 3.426814)]
demand = list(range(21))

mdp = InventoryMDP()

# Cost structures and days of the week
# c = [[10, 1, 20, 2], [10, 1, 20, 5], [10, 1, 20, 20], [20, 1, 20, 2], [20, 1, 20, 5]]
c = [[20, 1, 20, 20]]
c01 = [0, 1, 2, 3, 4, 5, 6]

# Main execution block
if __name__ ==  '__main__':
    pool = mp.Pool(processes=40)
    results = [pool.apply_async(implementation, args=(x, y)) for x in c for y in c01]
    output = [p.get() for p in results]
    open_file = open("InfoRelaxRandHorizonLength_m5_Rep100.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()