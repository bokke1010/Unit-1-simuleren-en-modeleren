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
    succes, x, y = simulate.simulate(y0, 1, alpha, beta, kappa, (4, t_end))
    infected = y[1]
    if not succes:
        raise ValueError("ivp couldn't be solved")
    return infected - real

def get_data():
    real_data = data_parser.readData()
    real_hosp = data_parser.extract_interval(real_data, date(2020, 3, 1), date(2020, 6, 27))
    real_hosp_comm = data_parser.compute_average(real_hosp, data_parser.HOSPITALIZATION_TIME)
    
    return real_hosp_comm / data_parser.FRACTION_HOSPITALIZED

intial_params = (alpha, beta)
# data = {
#     "y0": (597000, 3000, 0),
#     "t_end": 121,
#     "kappa": np.array([
#         [1]
#     ]),
#     "real": get_data()
# }
data = [(597000, 3000, 0), 121, np.array([[1]]), get_data()]
found_params = scipy.optimize.least_squares(fun_wrapper, intial_params, args = data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    found_alpha, found_beta = found_params['x']
    suc, fx, fy = simulate.simulate(data[0], 1, found_alpha, found_beta, data[2], (4, data[1]))
    # data_parser.plot_hosp(data[3], fy[1], "Real vs simulated data")
    plt.plot(fx, fy[0])
    plt.plot(fx, fy[1])
    plt.plot(fx, fy[2])
    plt.show()
