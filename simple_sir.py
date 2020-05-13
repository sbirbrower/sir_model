#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# func: a function that should take in a list of [y, y', y'', ...] and output derivatives at that point as a numpy array
def runge_kutta(func, t_i, t_f, initial_cond, h):
    X = [t_i]
    Y = [initial_cond]
    y = initial_cond
    while t_i < t_f:
        h = min(h, t_f - t_i)

        K0 = h * func(t_i, y)
        K1 = h * func(t_i + h / 2, y + K0 / 2)
        K2 = h * func(t_i + h / 2, y + K1 / 2)
        K3 = h * func(t_i + h, y + K2)
        
        # average the slopes, weighing half steps more
        y = y + (K0 + 2 * K1 + 2 * K2 + K3)/ 6
        t_i = t_i + h
        X.append(t_i)
        Y.append(y)
    return np.array(X), np.array(Y)


class SIR_model:

    def __init__(self):
        self.gamma = 0.1     # recovery rate (days^-1), estimate for covid-19
        self.contacts = 10
        self.prob = 0.019     # prob of transmission in a contact between an S and an I person
        self.beta = self.contacts * self.prob     # 0.5 transmission rate
        self.pop = 10000      # population size
        self.I0 = 1           # initial infected
        self.S0 = self.pop - self.I0
        self.R0 = 0
        self.initial_conds = np.array([self.S0/self.pop, self.I0/self.pop, self.R0/self.pop])

    def changeTransRate(self, transmission):
        self.beta = transmission
        
    def changeRecovRate(self, recovery):
        # percentage increase in recovery rate (i.e. new drug makes 30% more likely to recover)
        self.gamma = self.gamma * (1 + recovery)

    def changePopSize(self, size):
        self.pop = size
        self.S0 = self.pop - self.I0
    
    def changeInitInfect(self, num):
        self.I0 = num
        self.S0 = self.pop - self.I0

    def increaseNumContactsBy(self, contact):
        # how many more contacts per day does the infectious person have
        self.contacts = self.contacts + contact
        self.beta = self.contacts * self.prob
    
    def increaseDiseaseJumpRateBy(self, percent):
        # percentage increase in the risk per contact change (i.e. more people are wearing masks)
        self.prob = self.prob * (1 + percent)
        self.beta = self.prob * self.contacts

    def changePercentPopInateImmune(self, percentage):
        total_immune = (self.pop / 100) * percentage
        self.R0 = total_immune
        self.pop = self.pop - self.R0

    def dydt(self, x, y):
        # input: [S, I, R]
        # output: [dSdt, dIdt, dRdt]
        arr = np.zeros(3)
        arr[0] = y[0] * -self.beta * y[1]    # dSdt
        arr[1] = y[1] * self.beta * y[0] - self.gamma * y[1]  # dSdt
        arr[2] = self.gamma * y[1]     # dRdt
        return arr

    def restore_defaults(self):
        self.gamma = 0.25     # recovery rate (days^-1)
        self.contacts = 20
        self.prob = 0.025     # prob of transmission in a contact between an S and an I person
        self.beta = self.contacts * self.prob     # 0.5 transmission rate
        self.pop = 10000      # population size
        self.I0 = 1           # initial infected
        self.S0 = self.pop - self.I0
        self.R0 = 0
        
    def simulate_country(self, country):
        # source: https://arxiv.org/pdf/2003.11221.pdf
        beta_per_country = {"Australia": 0.29, "Austria": 0.29, "Belgium": 0.27,
                           "Brazil": 0.37, "Canada": 0.33, "Chile": 0.37, 
                            "China": 0.0012, "Czechia": 0.29, "Denmark": 0.12, 
                            "Ecuador": 0.48, "France": 0.24, "Germany": 0.28, 
                            "Iran": 0.11, "Ireland": 0.35, "Israel": 0.3, 
                            "Italy": 0.19, "Japan": 0.077, "South Korea": 0.02, 
                            "Luxembourg": 0.42, "Malaysia": 0.26, "Netherlands": 0.25,
                            "Norway": 0.15, "Pakistan": 0.31, "Poland": 0.31, 
                            "Spain": 0.28, "Sweden": 0.15, "Switzerland": 0.28, 
                            "United States": 0.38, "United Kingdom": 0.29
                           }
        country_pop = {"Australia": 21515754, "Austria": 8205000, "Belgium": 10403000,
                           "Brazil": 201103330, "Canada": 33679000, "Chile": 16746491, 
                            "China": 1330044000, "Czechia": 10476000, "Denmark": 5484000, 
                            "Ecuador": 14790608, "France": 64768389, "Germany": 0.28, 
                            "Iran": 76923300, "Ireland": 4622917, "Israel": 7353985, 
                            "Italy": 60340328, "Japan": 127288000, "South Korea": 48422644, 
                            "Luxembourg": 497538, "Malaysia": 28274729, "Netherlands": 16645000,
                            "Norway": 5009150, "Pakistan": 184404791, "Poland": 38500000, 
                            "Spain": 46505963, "Sweden": 9828655, "Switzerland": 7581000, 
                            "United States": 310232863, "United Kingdom": 62348447
                           }

        current_cases = {"Australia": 701, "Austria": 1290, "Belgium": 30604,
                           "Brazil": 83720, "Canada": 31760, "Chile": 14248, 
                            "China": 148, "Czechia": 3372, "Denmark": 1700, 
                            "Ecuador": 23921, "France": 94310, "Germany": 19375, 
                            "Iran": 14567, "Ireland": 4204, "Israel": 4831, 
                            "Italy": 84842, "Japan": 9150, "South Korea": 1008, 
                            "Luxembourg": 226, "Malaysia": 1552, "Netherlands": 36710,
                            "Norway": 7848, "Pakistan": 20803, "Poland": 9429, 
                            "Spain": 63148, "Sweden": 17730, "Switzerland": 2021, 
                            "United States": 1029198, "United Kingdom": 183329
                           }
        try:
            self.beta = beta_per_country[country]
            self.pop = country_pop[country]
            self.I0 = current_cases[country]
            
            # not all countries keep data on recovered, so assume none recovered
            self.S0 = self.pop - self.I0
            self.initial_conds = np.array([self.S0/self.pop, self.I0/self.pop, self.R0/self.pop])
        except KeyError:
            print("No data on that country")
        
        
    def plot_model(self, t_1, t_2):
        X, Y = runge_kutta(self.dydt, t_1, t_2, self.initial_conds, h=1)
        plt.plot(np.transpose(Y)[0], label="susceptible")
        plt.plot(np.transpose(Y)[1], label="infected")
        plt.plot(np.transpose(Y)[2], label="recovered")
        plt.legend(loc="upper right")
        plt.title("SIR model")
        plt.show()