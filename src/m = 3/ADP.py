import pandas as pd
import numpy as np
from scipy import stats
import math
import ast
import pickle
from scipy.special import gamma, factorial
from sklearn.linear_model import LinearRegression
import multiprocessing as mp

# Function to calculate the 95% confidence interval
def mean_confidence_interval_95(data):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = 1.96*se
    return m, m-h, m+h, h

# Function to generate a random variable following a multinomial distribution
def RV_Multinomial(n, p, U):
    X = np.zeros(len(p), dtype = int)
    for i in range(n):
        for j in range(len(p)):
            if U[i] <= sum(p[k] for k in range(j+1)):
                X[j] = X[j] + 1
                break
    return(X)

# Class definition for a non-perishable inventory Markov Decision Process
class NonPerishableInventoryMDP(object):
    def startState(self):
        # Initial state (e.g., zero inventory)
        return 0

    def orders(self, state):
        # Returns list of valid actions (possible order sizes)
        TopUp = list(range(21))
        result = [max(x - state, 0) for x in TopUp]
        return list(set(result))

    def succProbCost(self, state, order, demand, size, prob, day, cs):
        # Compute the successor states, probabilities, and associated costs
        result = []
        for i in range(len(demand)):
            # Update state after demand
            newState = max(order - max(demand[i] - state, 0), 0)
            # Cost calculation:
            # cs[0]: fixed ordering cost
            # cs[1]: holding cost
            # cs[2]: shortage cost
            cost = cs[0] * int(order > 0) + cs[1] * max(order + state - demand[i], 0) + cs[2] * max(demand[i] - order - state, 0)
            # Compute demand probabilities
            if (demand[i] == 20):
                dprob = 1 - sum(gamma(i + size[day]) * (prob[day] ** size[day]) * (1 - prob[day]) ** i / (gamma(size[day]) * factorial(i)) for i in range(20))
                result.append((newState, dprob, cost))
            else:
                dprob = gamma(demand[i] + size[day]) * (prob[day] ** size[day]) * (1 - prob[day]) ** demand[i] / (gamma(size[day]) * factorial(demand[i]))
                result.append((newState, dprob, cost))
        return result

    def discount(self):
        # Discount factor for future costs
        return 0.95

    def states(self):
        # List of all possible states (inventory levels)
        return [i for i in range(21)]

# Function to perform value iteration for the Non-Perishable Inventory MDP
def NonPerishableValueIteration(mdp, demand, size, prob, cs):
    # Initialize value function for each state and day
    V = [{} for _ in range(7)]  # 7 days (one for each day of the week)
    for i in range(7):
        for state in mdp.states():
            V[i][str(state)] = 0.0

    def Q(state, order, demand, day):
        # Compute the expected cost for a given state, action, and day
        return sum(dprob * (cost + mdp.discount() * V[(day + 1) % 7][str(newState)])
                   for newState, dprob, cost in mdp.succProbCost(state, order, demand, size, prob, day, cs))

    day = 0
    # Initialize new values to store updated state values during iteration
    newV = [{} for _ in range(7)]
    for i in range(7):
        for state in mdp.states():
            newV[i][str(state)] = 0.0

    while True:

        for state in mdp.states():
            # Update value for each state by minimizing over all possible actions
            newV[day][str(state)] = min(Q(state, order, demand, day) for order in mdp.orders(state))

        # Check for convergence
        if (day == 6):
            if max(abs(V[day][str(state)] - newV[day][str(state)]) for day in range(7) for state in mdp.states()) < 0.1:
                break
            else:
                V = newV
                day = 0
                # Reinitialize newV for the next iteration
                newV = [{} for _ in range(7)]
                for i in range(7):
                    for state in mdp.states():
                        newV[i][str(state)] = 0.0
        else:
            day += 1

    return V

# Class for MDP with perishable inventory
class InventoryMDP(object):
    def orders(self, state):
        TopUp = list(range(21))
        result = [max(x - sum(state), 0) for x in TopUp]
        return list(set(result))
    def succProbCost(self, state, order, demand, size, prob, day, cs):
        c02, c03 = c0
        c12, c13 = c1
        P2 = np.exp(c02 + c12 * order)/(1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
        P3 = np.exp(c03 + c13 * order)/(1 + np.exp(c02 + c12 * order) + np.exp(c03 + c13 * order))
        P1 = 1 - P2 - P3
        result = []
        for i in range(len(demand)):
            for y3 in range(order+1):
                for y2 in range(order-y3+1):
                    y1 = order - y3 - y2
                    newState = [0, 0]
                    newState[1] = max(state[0] + y2 - max(demand[i] - state[1] - y1, 0), 0)
                    newState[0] = max(y3 - max(demand[i] - state[0] - y2 - state[1] - y1, 0), 0)
                    cost = cs[0]*int(order>0) + cs[1]*max(order + state[0] + state[1] - demand[i], 0) + cs[2]*max(demand[i] - order - state[0] - state[1], 0) + cs[3]*max(state[1] + y1 - demand[i], 0)
                    aprob = math.factorial(order)*pow(P1,y1)*pow(P2,y2)*pow(P3,y3)/(math.factorial(y3)*math.factorial(y2)*math.factorial(y1))
                    if (demand[i] == 20):
                        dprob = 1-sum(gamma(i+size[day])*(prob[day]**size[day])*(1-prob[day])**i/(gamma(size[day])*factorial(i)) for i in range(20))
                        result.append((newState, dprob, aprob, cost))
                    else:
                        dprob = gamma(demand[i]+size[day])*(prob[day]**size[day])*(1-prob[day])**demand[i]/(gamma(size[day])*factorial(demand[i]))
                        result.append((newState, dprob, aprob, cost))
        return result
    def discount(self):
        return 0.95

# Function to calculate the expected approximate cost
def Q(state, order, day, b, cs, demand=list(range(21))):
    ExpVal = 0
    for newState, dprob, aprob, cost in mdp1.succProbCost(state, order, demand, size, prob, day, cs):
        if str(newState) not in Vhat[(day + 1) % 7]:
            data = [1]
            data.append(VF.iloc[(day + 1) % 7][sum(newState)])
            data.append(newState[0])
            data.append(newState[1])
            data.append(newState[0] ** 2)
            data.append(newState[1] ** 2)
            Vhat[(day + 1) % 7][str(newState)] = np.matmul(np.array(data), np.array(b[(day + 1) % 7]))
        ExpVal += dprob * aprob * (cost + mdp1.discount() * Vhat[(day + 1) % 7][str(newState)])
    return ExpVal

# Function to determine order decision with the expected approximate cost
def Decision(dow, s, b, cs):
    return min((Q(s, order, dow, b, cs), order) for order in mdp1.orders(s))[1]

# Function for exploration phase to gather state trajectories
def Exploration(cs, replication=30, T=100):
    rng3 = np.random.RandomState(seed=3)  # For demand generation
    rng4 = np.random.RandomState(seed=4)  # For shelf-life generation
    rng5 = np.random.RandomState(seed=5)  # For initial day and state

    U = rng4.random((replication, T, 20)) # Random samples for multinomial distribution

    samples = [[] for day in range(7)]  # Record states observed for each day of the week
    for rep in range(replication):
        demand = list()
        sd = rng5.choice([i for i in range(7)])  # Starting day
        # Generate demand sequence
        for t in range(T):
            dem = rng3.negative_binomial(size[(t + sd) % 7], prob[(t + sd) % 7])
            if dem > 20:
                demand.append(20)
            else:
                demand.append(dem)

        state = []
        for i in range(m - 1):
            state.append(rng5.choice([j for j in range(21 - sum(state))]))

        sq = [set() for day in range(7)]  # Unique states observed

        for t in range(T):
            sq[(t + sd) % 7].add(str(state))
            # Determine the order size
            if str(state) in pi[(t + sd) % 7]:
                order = pi[(t + sd) % 7][str(state)]
            else:
                order = Decision((t + sd) % 7, state, b, cs)
                pi[(t + sd) % 7][str(state)] = order

            P = []
            for i in range(m):
                if i < m - 1:
                    P.append(
                        np.exp(c0[i] + c1[i] * order) / (1 + sum(np.exp(c0[j] + c1[j] * order) for j in range(m - 1))))
                else:
                    P.insert(0, 1 - sum(P))
            y = RV_Multinomial(order, P, U[rep][t])

            # Update the inventory state
            newState = [0 for i in range(m - 1)]
            for j in range(m - 2):
                newState[m - 2 - j] = max(
                    state[m - 3 - j] + y[j + 1] - max(demand[t] - sum(state[m - 2 - k] + y[k] for k in range(j + 1)),
                                                      0), 0)
            newState[0] = max(y[m - 1] - max(demand[t] - sum(state[m - 2 - k] + y[k] for k in range(m - 1)), 0), 0)
            state = newState

        for day in range(7):
            samples[day].append(list(sq[day]))

    return samples

# Function for exploitation phase to refine policy using regression
def Exploitation(traj, cs, replication=30, T=100, alpha=0.95):
    IniState = [[] for day in range(7)]
    result = [[] for day in range(7)]
    b1 = [[] for day in range(7)]
    for day in range(7):
        IniState[day] = list(set().union(*traj[day]))
        for s in IniState[day]:
            rng3 = np.random.RandomState(seed=3)
            rng4 = np.random.RandomState(seed=4)

            U = rng4.random((replication, T, 20))

            x_list = []
            for rep in range(replication):
                demand = list()
                for t in range(T):
                    dem = rng3.negative_binomial(size[(t + day) % 7], prob[(t + day) % 7])
                    if dem > 20:
                        demand.append(20)
                    else:
                        demand.append(dem)

                state = ast.literal_eval(s)
                cost = 0
                for t in range(T):

                    if str(state) in pi[(t + day) % 7]:
                        order = pi[(t + day) % 7][str(state)]
                    else:
                        order = Decision((t + day) % 7, state, b, cs)
                        pi[(t + day) % 7][str(state)] = order

                    P = []
                    for i in range(m):
                        if i < m - 1:
                            P.append(np.exp(c0[i] + c1[i] * order) / (
                                        1 + sum(np.exp(c0[j] + c1[j] * order) for j in range(m - 1))))
                        else:
                            P.insert(0, 1 - sum(P))
                    y = RV_Multinomial(order, P, U[rep][t])

                    # Update the inventory state
                    newState = [0 for i in range(m - 1)]
                    for j in range(m - 2):
                        newState[m - 2 - j] = max(state[m - 3 - j] + y[j + 1] - max(
                            demand[t] - sum(state[m - 2 - k] + y[k] for k in range(j + 1)), 0), 0)
                    newState[0] = max(y[m - 1] - max(demand[t] - sum(state[m - 2 - k] + y[k] for k in range(m - 1)), 0),
                                      0)

                    cost += pow(alpha, t) * (
                                cs[0] * int(order > 0) + cs[1] * max(order + sum(state) - demand[t], 0) + cs[2] * max(
                            demand[t] - order - sum(state), 0) + cs[3] * max(state[m - 2] + y[0] - demand[t], 0))
                    state = newState

                x_list.append(cost)

            result[day].append(np.mean(x_list))

        # Features and targets for regression
        X, y = [], []
        for s in range(len(IniState[day])):
            data = [1]
            data.append(VF.iloc[day][sum(ast.literal_eval(IniState[day][s]))])
            data.append(ast.literal_eval(IniState[day][s])[0])
            data.append(ast.literal_eval(IniState[day][s])[1])
            data.append(ast.literal_eval(IniState[day][s])[0] ** 2)
            data.append(ast.literal_eval(IniState[day][s])[1] ** 2)
            X.append(np.array(data))
            y.append(result[day][s])

        reg = LinearRegression(fit_intercept=False).fit(np.array(X), np.array(y))
        b1[day] = list(reg.coef_)

    return b1

# Function to compute expected costs
def ExpCost(pi, cs, alpha=0.95, replication=1000):
    result = []
    rng6 = np.random.RandomState(seed=5)
    rng7 = np.random.RandomState(seed=6)

    U = rng7.random((replication, 100, 20))

    for rep in range(replication):
        demand = list()
        for t in range(100):
            dem = rng6.negative_binomial(size[t % 7], prob[t % 7])
            if dem > 20:
                demand.append(20)
            else:
                demand.append(dem)

        state = [0 for i in range(m - 1)]
        cost = 0
        for t in range(100):
            day = t % 7
            if str(str(state)) in pi[day]:
                order = pi[day][str(state)]
            else:
                order = Decision(day, state, b, cs)
                pi[day][str(state)] = order

            P = []
            for i in range(m):
                if i < m - 1:
                    P.append(
                        np.exp(c0[i] + c1[i] * order) / (1 + sum(np.exp(c0[j] + c1[j] * order) for j in range(m - 1))))
                else:
                    P.insert(0, 1 - sum(P))
            y = RV_Multinomial(order, P, U[rep][t])

            # Update the inventory state
            newState = [0 for i in range(m - 1)]
            for j in range(m - 2):
                newState[m - 2 - j] = max(
                    state[m - 3 - j] + y[j + 1] - max(demand[t] - sum(state[m - 2 - k] + y[k] for k in range(j + 1)),
                                                      0), 0)
            newState[0] = max(y[m - 1] - max(demand[t] - sum(state[m - 2 - k] + y[k] for k in range(m - 1)), 0), 0)
            cost += pow(alpha, t) * (
                        cs[0] * int(order > 0) + cs[1] * max(order + sum(state) - demand[t], 0) + cs[2] * max(
                    demand[t] - order - sum(state), 0) + cs[3] * max(state[m - 2] + y[0] - demand[t], 0))
            state = newState

        result.append(cost)

    return result


def implementation(cs, Plogit):
    global Vhat, pi, b, VF, m, c0, c1, size, prob, mdp1
    # Maximum shelf-life of three days
    m = 3
    # Parameters for multinomial logit model
    c0 = [Plogit[0], Plogit[1]]
    c1 = [Plogit[2], Plogit[3]]
    # Parameters for negative-binomial demand distribution
    size = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
    prob = [size[0] / (size[0] + 5.660569), size[1] / (size[1] + 6.922555), size[2] / (size[2] + 6.504332),
            size[3] / (size[3] + 6.165049), size[4] / (size[4] + 5.816060), size[5] / (size[5] + 3.326408),
            size[6] / (size[6] + 3.426814)]
    d = list(range(21))   # Demand range

    # Compute value function for non-perishable inventory problem
    mdp = NonPerishableInventoryMDP()
    VFDic = NonPerishableValueIteration(mdp, demand=d, size=size, prob=prob, cs=cs)
    VF = pd.DataFrame.from_dict(VFDic, orient='columns')    # Value function table

    # Initialize MDP for the perishable inventory problem
    mdp1 = InventoryMDP()
    iteration = 20  # Number of iterations for the ADP algorithm
    Cost_list = []  # Track cost progression over iterations

    # Initialize policy (pi) and value function (Vhat)
    b = [[0 for _ in range(6)] for _ in range(7)]  # Regression coefficients
    pi = [{} for _ in range(7)]  # Policy for each day of the week
    Vhat = [{} for _ in range(7)]  # Value function approximation

    # Compute initial cost (myopic policy)
    Cost_list.append(ExpCost(pi, cs))

    # Iterative ADP loop
    for i in range(iteration):
        # Exploration phase: Collect state-action trajectories
        trajs = Exploration(cs)
        # Exploitation phase: Update regression coefficients using collected data
        b1 = Exploitation(trajs, cs)
        # Blend new coefficients with previous ones using a learning rate
        if i > 0:
            lambd = 1 / (i + 1)
            for dow in range(7):    # Dow: Day of week
                b1[dow] = [(1 - lambd) * b[dow][k] + lambd * b1[dow][k] for k in range(len(b[dow]))]

        b = b1  # Update coefficients
        pi = [{} for _ in range(7)]  # Reset policy
        Vhat = [{} for _ in range(7)]  # Reset value function
        Cost_list.append(ExpCost(pi, cs))  # Record cost for this iteration

    # Organize results into a dictionary
    Dic = {'CS': [str(cs) for i in range(21)],
           'Endog.': [str(Plogit) for i in range(21)],
           'Iteration': [i for i in range(21)],
           'Cost': Cost_list}

    return (Dic)


# Cost structures and logit model parameters
c = [[10, 1, 20, 5], [10, 1, 20, 20], [10, 1, 20, 80], [100, 1, 20, 5], [100, 1, 20, 20],[100, 1, 20, 80]]
c01 = [[1, 0.5, -0.4, -0.8], [1, 0.5, -0.2, -0.1], [1, 0.5, -0.1, -0.05], [1, 0.5, 0.0, 0.0], [1, 0.5, 0.2, 0.4], [1, 0.5, 0.4, 0.8]]

if __name__ ==  '__main__':
    # Parallel computation of results for different parameter combinations
    pool = mp.Pool(processes=40)
    results = [pool.apply_async(implementation, args=(x, y)) for x in c for y in c01]
    # Collect results from parallel processes
    output = [p.get() for p in results]
    # Save results to a pickle file
    open_file = open("ADPerf_ExcMyp_Final.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()
