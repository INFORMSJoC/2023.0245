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
    X = np.zeros(len(p), dtype = int)   # Initialize counts for each category
    for i in range(n):  # Iterate over trials
        for j in range(len(p)): # Determine which category the trial falls into
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
            list: Initial state with no inventory ([0, 0]).
        """
        return [0, 0]

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
            y1, y2, y3 = sample[order]
            newState = [0, 0]
            # Update inventory states based on demand and order
            newState[1] = max(state[0] + y2 - max(demand[i] - state[1] - y1, 0), 0)
            newState[0] = max(y3 - max(demand[i] - state[0] - y2 - state[1] - y1, 0), 0)
            # Compute cost based on ordering, holding, shortage, and wastage costs
            cost = cs[0] * int(order > 0) + cs[1] * max(order + state[0] + state[1] - demand[i], 0) + cs[2] * max(demand[i] - order - state[0] - state[1], 0) + cs[3] * max(state[1] + y1 - demand[i], 0)
            # Compute demand probabilities
            if (demand[i] == 20):
                dprob = 1 - sum(gamma(i + size[day]) * (prob[day] ** size[day]) * (1 - prob[day]) ** i / (gamma(size[day]) * factorial(i)) for i in range(20))
                result.append((newState, dprob, cost))
            else:
                dprob = gamma(demand[i] + size[day]) * (prob[day] ** size[day]) * (1 - prob[day]) ** demand[i] / (gamma(size[day]) * factorial(demand[i]))
                result.append((newState, dprob, cost))
        return result

    def states(self):
        """
         Generate all possible inventory states.

         Returns:
             list: All feasible inventory states.
         """
        return [[i, j] for i in range(21) for j in range(21 - i)]

# Perform backward induction for value computation
def BackwardInduction(cs, T, sample):
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
        day = t%7   # Determine day of the week
        for state in mdp.states():
            newV[str(state)] = min(Q(state, order, day, t) for order in mdp.orders(state))
        V = newV
        t -= 1
    return V

# Main implementation function
def implementation(cs, Plogit):
    """
    Simulate the perishable inventory problem and compute lower-bound.

    Args:
        cs (list): Cost structure.
        Plogit (list): Logit model parameters for perishability.

    Returns:
        dict: Dictionary containing lower-bound for all inventory states at day 0.
    """
    rng1 = np.random.RandomState(seed=1)
    rng2 = np.random.RandomState(seed=2)
    Replication = 4000  # Number of replications
    length = rng1.geometric(0.05, size=Replication)     # Random horizon lengths
    U = [rng2.random((length[i], 20)) for i in range(Replication)]  # Random samples
    c02, c03, c12, c13 = Plogit
    LBV_list= []
    sample_list =[]
    # Generate shelf-life samples for each replication
    for rep in range(Replication):
        sample_path = []
        for h in range (length[rep]):
            sample = []
            for i in range(21):
                order = i
                P2 = np.exp(c02 + c12 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
                P3 = np.exp(c03 + c13 * order) / (1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
                P1 = 1 - P2 - P3
                sample.append(RV_Multinomial(i, [P1, P2, P3], U[rep][h]))
            sample_path.append(sample)
        sample_list.append(sample_path)

        # Perform backward induction to compute lower-bound
        LBV = BackwardInduction(cs=cs, T= length[rep], sample=sample_path)
        LBV_list.append(LBV)

    Dic = {'LBV': LBV_list}
    return (Dic)

# Define maximum shelf-life and daily demand distributions
m = 3
size = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
prob = [size[0] / (size[0] + 5.660569), size[1] / (size[1] + 6.922555), size[2] / (size[2] + 6.504332),
        size[3] / (size[3] + 6.165049), size[4] / (size[4] + 5.816060), size[5] / (size[5] + 3.326408),
        size[6] / (size[6] + 3.426814)]
demand = list(range(21))

mdp = InventoryMDP()

# Parameter grids for experiments
c = [[10, 1, 20, 5], [10, 1, 20, 20], [10, 1, 20, 80], [100, 1, 20, 5], [100, 1, 20, 20],[100, 1, 20, 80]]
c01 = [[1, 0.5, -0.4, -0.8], [1, 0.5, -0.2, -0.1], [1, 0.5, -0.1, -0.05], [1, 0.5, 0.0, 0.0], [1, 0.5, 0.2, 0.4], [1, 0.5, 0.4, 0.8]]

if __name__ ==  '__main__':
    # Parallelize computation across parameter combinations
    pool = mp.Pool(processes=40)
    results = [pool.apply_async(implementation, args=(x, y)) for x in c for y in c01]
    output = [p.get() for p in results]
    # Save results to a pickle file
    open_file = open("InfoRelaxRandHorizonLength_4000rep.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()
