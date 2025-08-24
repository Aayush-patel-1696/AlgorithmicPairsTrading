
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
        # Calculate loss returns 
   
        money_arr_arr = np.array(money_arr)

        # Calculate returns
        returns = np.diff(money_arr_arr)
        # Fill NaN values with 0
        returns = np.nan_to_num(returns)

        # Calculate mean and semi-deviation
        mean_returns = np.mean(returns)
        semi_deviation = np.mean(np.maximum(mean_returns-returns, 0))

        return mean_returns - self.chi*semi_deviation


class TotalSemiDeviation(RiskMeasure):

    def __init__(self, chi=0.4):
        self.chi = chi
    
    def calculate(self, money_arr):
        """
        Calculate the drawdown of a money array
        """
        # Convert to numpy array
        money_arr = np.array(money_arr)

        # Calculate returns
        money_diff_rtn = np.diff(money_arr)

        # Fill NaN values with 0
        money_diff_rtn = np.nan_to_num(money_diff_rtn)
        
       # Calculate drawdown
        money_diff_rtn_dv = np.maximum(np.mean(money_diff_rtn)-money_diff_rtn,0)

        return np.sum(money_diff_rtn) - self.chi*np.sum(money_diff_rtn_dv)
    

