

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd



class RiskMeasure(ABC):

    @abstractmethod
    def calculate(self, money_arr):
        pass


class CumulativeReturn(RiskMeasure):

    def __init__(self):
        pass

    def calculate(self, money_arr):
        return money_arr[-1]
    

class ExponationalUtility(RiskMeasure):

    def __init__(self, risk_aversion=0.01):
        self.risk_aversion = risk_aversion

    def calculate(self, money_arr):
        return -1*np.mean(np.exp(-self.risk_aversion *np.array(money_arr)))
    
class MeanSemiDeviation(RiskMeasure):

    def __init__(self,chi=0.3):
        self.chi = chi

    def calculate(self, money_arr):
        """
        Calcilate the mean semi deviation of a money array"
        rho = E[loss] + chi*E[[E[loss]-loss]_+]
        """
        money_arr = np.array(-1*money_arr)

        mean = np.mean(money_arr)
        semi_deviation = np.mean(np.maximum(mean - money_arr, 0))
        return mean + self.chi*semi_deviation