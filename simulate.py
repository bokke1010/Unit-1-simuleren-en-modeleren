import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def simulate_SIR(y0, districts, alpha, beta, kappa, t_int):

    def fun(t, y):
        S, I, R = np.split(y, 3)
        newInfected = beta * S * kappa.dot(I)
        return np.concatenate([- newInfected, newInfected - alpha * I, alpha * I])

    # Passing in max_step gives a smoother result, without it the graph would be all pointy
    # solution = solve_ivp(fun_wrapper, (0, 130), y0, dense_output=True, max_step = 0.2)
    solution = solve_ivp(fun, t_int, y0, dense_output=True, max_step = 0.2, t_eval=range(t_int[0], t_int[1] + 1))
    # print("ivp_done")

    return solution['success'], solution['t'], solution['y']

def simulate_SCIR(y0, districts, alpha, beta, gamma, kappa, t_int):
    def fun(t, y):
        S, C, I, R = np.split(y, 4)
        contact_vector = kappa.dot(I)
        S_Infected = beta * S * contact_vector
        C_Infected = gamma * C * contact_vector
        return np.concatenate([- S_Infected, - C_Infected, S_Infected + C_Infected - alpha * I, alpha * I])

    # Passing in max_step gives a smoother result, without it the graph would be all pointy
    # solution = solve_ivp(fun_wrapper, (0, 130), y0, dense_output=True, max_step = 0.2)
    solution = solve_ivp(fun, t_int, y0, dense_output=True, max_step = 0.2, t_eval=range(t_int[0], t_int[1] + 1))
    # print("ivp_done")

    return solution['success'], solution['t'], solution['y']

if __name__ == "__main__":
    
    # Number of idependent areas
    districts = 1
    # Contact rate
    beta = 0.0000009
    gamma = 0.000000004
    # Rate of removal
    alpha = 0.16
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
    y0 = (360000, 16860000, 1, 0)

    succes, t, y = simulate_SCIR(y0, districts, alpha, beta, gamma, kappa, (0, 120))
    plt.plot(t, y[0], label="ES")
    plt.plot(t, y[1], label="CF")
    plt.plot(t, y[2], label="I")
    plt.plot(t, y[3], label="R")
    plt.legend()
    plt.show()
    print(y[3][-1])

    
    # if verbose:
    #     print(solution["message"])

    #     messages = ["Susceptible ", "Infected ", "Removed "]
    #     for i in range(len(solution['y'])):
    #         subgroup, group = i//districts, i % districts
    #         plt.plot(solution['t'], solution['y'][i], label = messages[subgroup] + str(group))
    #     plt.legend()
    #     plt.show()