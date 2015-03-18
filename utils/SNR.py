import numpy as np

def SNR(self, S, S0):
    """
    S - actual value
    S0 - estimated value
    
    Returns:
    Signal to noise ratio
    """
    return np.var(S) / np.var(S - S0)
