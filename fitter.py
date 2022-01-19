import scipy.optimize
import numpy as np
from datetime import date

import simulate
import data_parser

districts = 1
# Contact rate
beta = 0.000001
# Rate of removal
alpha = 0.14
# Transfer from district with index row to disctrict with index column coefficient, should be 1 within a region (on the diagonal)

def fun_wrapper(params, y0, t_end, kappa, real):
    alpha, beta = params
    # real_infected, real_susceptible = real
    succes, x, y = simulate.simulate(y0, 1, alpha, beta, kappa, (0, t_end))
    susceptible, infected, removed = y
    if not succes:
        raise ValueError("ivp couldn't be solved")
    return infected - real
    # return np.concatenate((infected - real_infected, susceptible - real_susceptible))

def get_data():
    real_data = data_parser.readData()
    real_hosp = data_parser.extract_interval(real_data, date(2020, 3, 1), date(2020, 6, 27))
    real_hosp_curr = data_parser.compute_average(real_hosp, data_parser.HOSPITALIZATION_TIME)
    # Experimented with running least_squares on both infection and susceptible data, but that did not
    # result in any sufficiently accurate results
    # real_hosp_comm = np.cumsum(real_hosp / data_parser.FRACTION_HOSPITALIZED)
    
    return real_hosp_curr / data_parser.FRACTION_HOSPITALIZED#, data_parser.POPULATION_NL - real_hosp_comm

intial_params = (alpha, beta)
# data = {
#     "y0": (597000, 3000, 0),
#     "t_end": 121,
#     "kappa": np.array([
#         [1]
#     ]),
#     "real": get_data()
# }
real_world = get_data()
initial_infected = real_world[0]

# This model assumes the entire population eventually gets infected
# Due to this, the population size has to be roughly equal to the total infections
# total = data_parser.POPULATION_NL
total = 600000

const_params = [
    (total - initial_infected, initial_infected, 0),
    len(real_world) - 1,
    np.array([[1]]),
    real_world
]
found_params = scipy.optimize.least_squares(fun_wrapper, intial_params, args = const_params)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    found_alpha, found_beta = found_params['x']
    suc, fx, fy = simulate.simulate(const_params[0], 1, found_alpha, found_beta, const_params[2], (0, const_params[1]))
    # data_parser.plot_hosp(const_params[3], fy[1], "Real vs simulated const_params")
    plt.plot(fx, fy[0])
    plt.plot(fx, fy[1])
    plt.plot(fx, fy[2])
    plt.show()
