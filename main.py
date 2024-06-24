
from percdl import (
    TransformationFunction, PerCDL
)
import pandas as pd
import numpy as np
import os

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def main(input, L, D, W, i_s):

    # Load data
    df = pd.read_csv(input)
    X = df.to_numpy().T

    # Fixed algorithm parameters (just for the demo)
    K = 1
    n_steps = 15

    # Transformation function
    f = TransformationFunction(L, D, W)

    # Solver
    percdl = PerCDL(f, X, K, L, n_steps=n_steps)

    # Initialization
    Phi_init = np.zeros((K, L))
    Phi_init[0, :] = X[0, i_s:i_s+L]
    percdl.initialize(Phi=Phi_init)

    # Optimization
    percdl.run()

    # Plots
    percdl.plot_reconstruction()
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("-L", type=int, required=True)
    parser.add_argument("-D", type=int, required=True)
    parser.add_argument("-W", type=int, required=True)
    parser.add_argument("-i_s", type=int, required=False, default=0)

    args = parser.parse_args()
    main(args.input, args.L, args.D, args.W, args.i_s)
