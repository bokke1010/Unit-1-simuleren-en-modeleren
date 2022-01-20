import scipy.optimize
import numpy as np
from datetime import date

import simulate
import data_parser

districts = 1
# Contact rate
beta = 0.0000008
gamma = 0.000000003
# Rate of removal
alpha = 0.16
# Transfer from district with index row to disctrict with index column coefficient, should be 1 within a region (on the diagonal)

def fun_wrapper(params, y0, t_end, kappa, real):
    alpha, beta, gamma = params
    # real_infected, real_susceptible = real
    succes, x, y = simulate.simulate_SCIR(y0, 1, alpha, beta, gamma, kappa, (0, t_end))
    susceptible, careful, infected, removed = y
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

intial_params = (alpha, beta, gamma)
real_world = get_data()
initial_infected = real_world[0]

# This model assumes the entire population eventually gets infected
# Due to this, the population size has to be roughly equal to the total infections
not_infected = data_parser.POPULATION_NL - initial_infected
care_factor = 0.97

const_params = [
    (not_infected * (1-care_factor), not_infected * care_factor, initial_infected, 0),
    len(real_world) - 1,
    np.array([[1]]),
    real_world
]

found_params = scipy.optimize.least_squares(fun_wrapper, intial_params, args = const_params)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    found_alpha, found_beta, found_gamma = found_params['x']
    suc, fx, fy = simulate.simulate_SCIR(const_params[0], 1, found_alpha, found_beta, found_gamma, const_params[2], (0, const_params[1]))
    print(found_params['x'], fy[3][-1])
    data_parser.plot_diff("Real vs simulated const_params", ["simulated", "real"], const_params[3], fy[2])
    plt.plot(fx, fy[0])
    plt.plot(fx, fy[1])
    plt.plot(fx, fy[2])
    plt.show()
