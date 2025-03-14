

from abc import ABC, abstractmethod
import numpy as np



class RiskMeasure(ABC):

    @abstractmethod
    def calculate(self, money_arr):
        pass



class CumulativeReturn(RiskMeasure):

    def __init__(self):
        pass

    def calculate(self, money_arr):
        return money_arr[-1]
    

class MeanSemiDeviation(RiskMeasure):

    def __init__(self,chi):
        self.chi = chi

    def calculate(self, money_arr):
        mean_semideviation = np.mean(money_arr) - self.chi*(np.mean(np.abs(np.mean(money_arr)-money_arr)))
        return mean_semideviation