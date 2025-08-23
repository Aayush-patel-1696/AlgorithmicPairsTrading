"""Grid Search Optimization for pairs trading parameters"""

import itertools
import numpy as np
from sklearn.model_selection import ParameterGrid
from .utils import trade

class gridsearchOpt:

    def __init__(self, risk_measure):
        self.optimize_results = None
        self.risk_measure = risk_measure

    def optimize(self, beta, spread, S1, S2, param_grid):
        """
        Grid Search Optimization over parameter grid
        """
        best_score = -np.inf
        best_params = None
        all_results = []

        # Print the total length of parameters
        print("Total Parameters :", len(list(ParameterGrid(param_grid))))

        for i, params in enumerate(ParameterGrid(param_grid)):
            window1 = params["window1"]
            window2 = params["window2"]
            sell_threshold = params["sell_threshold"]
            buy_threshold = params["buy_threshold"]
            clear_threshold = params["clear_threshold"]

            money = trade(
                S1, S2, spread, beta, 
                window1, window2,
                sell_threshold, buy_threshold, clear_threshold
            )

            score = self.risk_measure.calculate(money)

            all_results.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                best_params = params

            #print progress in percentage
            if i%100==0:
                print(f"Current Progress: {i + 1}/{len(list(ParameterGrid(param_grid)))}")

        self.optimize_results = {"best_params": best_params, "best_score": best_score, "all_results": all_results}
        return self.optimize_results

    
