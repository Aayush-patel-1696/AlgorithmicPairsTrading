

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
    

