from datetime import date, timedelta
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

# Constants

# From https://www.volksgezondheidtoekomstverkenning.nl/c-vtv/covid-19/ziekte
FRACTION_HOSPITALIZED = 0.0185
# (0.35/1.85)*18.7 + (1.5/1.85)*8.2 = 10.1864864865
# Numbers from other sources
# Are we confident the hospitalization time is equal to the duration of the sickness? Do we need another constant for infected?
HOSPITALIZATION_TIME = 10.19

# From ??? (via google) (2020)
POPULATION_NL = 17440000
# From CBS (2013)
POPULATION_AMS = 981095

# From https://data.rivm.nl/covid-19/
filepath = "COVID-19_ziekenhuisopnames.csv"


def readData():
    return pd.read_csv(filepath, delimiter=';')

def filter_region(data, region):
    # Extract the data for security region VR13, Amsterdam. 
    Data_Amsterdam = data[data['Security_region_code'] == region]
    # Extract the data for the rest of the Netherlands
    Data_rest = data[data['Security_region_code'] != region]

    return Data_Amsterdam, Data_rest

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def extract_interval(data, start_date: date, end_date: date):
    return np.array([sum((data[data['Date_of_statistics'] == str(single_date)])['Hospital_admission']) for single_date in daterange(start_date, end_date)])


def plot_diff(header, labels, *all_data, axis = ["Days", "People"]):
    days = np.arange(len(all_data[0]))
    plt.clf()
    for data in all_data:
        plt.plot(days, data)
    plt.title(header)
    plt.legend(labels)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.show()

def compute_average(data, columns):
    # Create a cumulative plot, average sickness duration 10 days.
    weight_length = ceil(columns)
    weights = np.ones(weight_length)
    if weight_length != columns:
        weights[0] = columns % 1
    return np.convolve(data, weights, 'same')

# Create the SIR plots. 

if __name__ == "__main__":
    data_ams, data_rest = filter_region(readData(), 'VR13')
    
    data_ams.describe()
    data_rest.describe()

    hosp_ams = extract_interval(data_ams, date(2020, 2, 27), date(2020, 6, 27))
    hosp_rest = extract_interval(data_rest, date(2020, 2, 27), date(2020, 6, 27))
    # plot_diff("Hospitalizations for Covid19", ['Rest of Netherlands', 'Amsterdam'], hosp_ams, hosp_rest)

    hosp_ams_comm, hosp_rest_comm = compute_average(hosp_ams, HOSPITALIZATION_TIME), compute_average(hosp_rest, HOSPITALIZATION_TIME)
    # plot_diff("Current hospitalizations for Covid19", ['Rest of Netherlands', 'Amsterdam'], hosp_ams_comm, hosp_rest_comm)

    inf_ams_comm, inf_rest_comm = hosp_ams_comm / FRACTION_HOSPITALIZED, hosp_rest_comm / FRACTION_HOSPITALIZED
    plot_diff("Infected people for Covid19", ['Rest of Netherlands', 'Amsterdam'], inf_ams_comm, inf_rest_comm)
    
    # pop_rest = POPULATION_NL - POPULATION_AMS


# Infected_rest_new = np.asarray(Rest_hosp) / 0.0185
# Infected_ams_new = np.asarray(Amsterdam_hosp) / 0.0185

# Inf_total_rest = np.cumsum(Infected_rest_new)
# Pop_rest_V = Inf_total_rest[-1]

# S_rest = np.ones(len(Inf_total_rest)) * Pop_rest_V  - Inf_total_rest
# R_rest = np.ones(len(Inf_total_rest)) * Pop_rest_V  - S_rest - Cumm_inf_rest[0:len(S_rest)]

# plt.plot(days, Cumm_inf_rest, S_rest)
# plt.plot(days[0:len(R_rest)], R_rest)
# plt.show()
