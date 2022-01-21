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

def fill_kappa(kappa, k_arr):
    x, y = 1, 0
    for item in k_arr:
        kappa[x][y] = item
        kappa[y][x] = item
        x += 1
        if x == len(kappa):
            y += 1
            x = y + 1
    return kappa

def fun_wrapper(params, y0, t_end, kappa, real):
    alpha, beta, gamma, *k_arr = params
    kappa = fill_kappa(kappa, k_arr)

    succes, x, y = simulate.simulate_SCIR(y0, 1, alpha, beta, gamma, kappa, (0, t_end))

    infected = y[2 * region_n : 3 * region_n]
    if not succes:
        raise ValueError("ivp couldn't be solved")

    return (infected - real).flatten()


def get_data(regions: list = []):
    region_data = [None] * (len(regions) + 1)
    region_data[0] = data_parser.readData()
    for i, region in enumerate(regions):
        region_data[i], region_data[i+1] = data_parser.filter_region(region_data[i], region)
    
    def to_curr_int(region):
        reg_hosp = data_parser.extract_interval(region, date(2020, 3, 1), date(2020, 6, 27))
        reg_hosp_curr = data_parser.compute_average(reg_hosp, data_parser.HOSPITALIZATION_TIME)
        return reg_hosp_curr / data_parser.FRACTION_HOSPITALIZED

    return list(map(to_curr_int, region_data))


real_world = get_data(['VR13'])
region_n, element_n = len(real_world), len(real_world[0])
kappa_par_size = (region_n * (region_n - 1)) // 2
intial_params = [alpha, beta, gamma] + ([1] * kappa_par_size)
print(intial_params)
care_factor = 0.97

# This model assumes the entire population eventually gets infected
# Due to this, the population size has to be roughly equal to the total infections
region_params = [0] * (4 * region_n)
POP = [data_parser.POPULATION_AMS, data_parser.POPULATION_NL - data_parser.POPULATION_AMS]
for i, data in enumerate(real_world):
    not_infected = POP[i] - data[0]
    region_params[i] = not_infected * (1-care_factor)
    region_params[i + region_n] = not_infected * care_factor
    region_params[i + 2 * region_n] = data[0]
print(region_params)
kappa = np.ones(shape=(region_n, region_n))
const_params = [
    region_params,
    element_n - 1,
    kappa,
    real_world
]

found_params = scipy.optimize.least_squares(fun_wrapper, intial_params, args = const_params)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    found_alpha, found_beta, found_gamma, *found_kappa = found_params['x']
    suc, fx, fy = simulate.simulate_SCIR(const_params[0], 1, found_alpha, found_beta, found_gamma, const_params[2], (0, const_params[1]))
    print(found_params['x'], fy[3][-1])
    for i in range(region_n):
        data_parser.plot_diff(f"Real vs simulated in region {i}", ["real", "simulated"], const_params[3][i], fy[2 * region_n + i])
    # plt.plot(fx, fy[0])
    # plt.plot(fx, fy[1])
    # plt.plot(fx, fy[2])
    # plt.show()
