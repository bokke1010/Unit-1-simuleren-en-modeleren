import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def fun(t, y, alpha, beta, kappa):
    S, I, R = np.split(y, 3)
    newInfected = beta * S * kappa.dot(I)
    return np.concatenate([- newInfected, newInfected - alpha * I, alpha * I])
    # return np.concatenate([gamma * R - newInfected, newInfected - alpha * I, alpha * I - gamma * R])

def simulate(y0, districts, alpha, beta, kappa, t_int, verbose : bool =False):

    def fun_wrapper(t, y):
        return fun(t, y, alpha, beta, kappa)

    # Passing in max_step gives a smoother result, without it the graph would be all pointy
    # solution = solve_ivp(fun_wrapper, (0, 130), y0, dense_output=True, max_step = 0.2)
    solution = solve_ivp(fun_wrapper, t_int, y0, dense_output=True, max_step = 0.2, t_eval=range(t_int[0], t_int[1] + 1))
    # print("ivp_done")

    if verbose:
        print(solution["message"])

        messages = ["Susceptible ", "Infected ", "Removed "]
        for i in range(len(solution['y'])):
            subgroup, group = i//districts, i % districts
            plt.plot(solution['t'], solution['y'][i], label = messages[subgroup] + str(group))
        plt.legend()
        plt.show()
    return solution['success'], solution['t'], solution['y']

if __name__ == "__main__":
    
    # Number of idependent areas
    districts = 1
    # Contact rate
    beta = 0.000001
    # Rate of removal
    alpha = 0.14
    # Return to susceptibility
    gamma = 0
    # Transfer from district with index row to disctrict with index column coefficient, should be 1 within a region (on the diagonal)
    # kappa = np.array([
    #     [1, 0.05, 0.01],
    #     [0.05, 1, 0.01],
    #     [0.01, 0.01, 1]
    # ])
    kappa = np.array([
        [1]
    ])
    # S0, S1, S2, I0, I1, I2, R0, R1, R2
    y0 = (600000, 1, 0)

    succes, t, y = simulate(y0, districts, alpha, beta, kappa, (0, 120), verbose=True)