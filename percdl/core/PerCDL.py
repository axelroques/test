
from ..optimization import (
    _CSC, _CDU, _CDU_perso, _IPU,
    normalize_Phi_Z, recenter_Phi, relearn_A
)
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np


class PerCDL:

    def __init__(
            self, f, X, K, L, 
            n_steps=100, step_size=1e-3, penalty=1e-2
        ) -> None:
        
        # Data parameters
        self.f = f
        self.X = X
        self.S = X.shape[0]
        self.N = X.shape[1]
        self.K = K
        self.L = L

        # Optimization parameters
        self.n_steps = n_steps
        self.step_size = step_size
        self.penalty = penalty

        # Initialize arrays of the correct shape with zeros
        self.reset()

    def reset(self):
        """
        Reset all arrays.
        """
        self.Phi_init = np.zeros((self.K, self.L))
        self.Z_init = np.zeros((self.S, self.K, self.N-self.L+1))
        self.A_init = np.zeros((self.S, self.K, self.f.M))
        self._initialized = False

    def initialize(self, Phi=None, Z=None, A=None):
        """
        Initialize arrays.
        """

        # Get initializations
        if Phi is not None:
            self.Phi_init = Phi
        if Z is not None:
            self.Z_init = Z
        if A is not None:
            self.A_init = A
        
        # Create copys
        self.Phi = self.Phi_init.copy()
        self.Z = self.Z_init.copy()
        self.A = self.A_init.copy()

        self._initialized = True

    def run(self):
        """
        Launch optimization process.
        """

        if not self._initialized:
            raise RuntimeError('PerCDL was not initialized.')
        
        # Initial estimates
        for _ in range(self.n_steps):
            
            # Normalization
            self.Phi, self.Z = normalize_Phi_Z(self.Phi, self.Z)
            
            # CSC
            Phi_perso = np.repeat(np.expand_dims(self.Phi, axis=1), self.S, axis=1)
            self.Z = self.CSC(Phi_perso)

            # CDU (no personalization)
            self.Phi = self.CDU()

        # Personalization
        for _ in range(self.n_steps):

            # Normalization
            self.Phi, self.Z = normalize_Phi_Z(self.Phi, self.Z)
            
            # CSC
            Phi_perso = self.f.personalize(self.Phi, self.A)
            self.Z = self.CSC(Phi_perso)

            # CDU (with personalization)
            Phi = self.CDU_perso() # This Phi may not be centred, we'll correct this below

            # IPU
            A = self.IPU(Phi) # This A may not be centred either

            # Barycenter step
            self.Phi = recenter_Phi(
                Phi, A, self.f.D, self.f.W, self.L
            )
            self.A = relearn_A(
                self.Phi, Phi, A, self.f.D, self.f.W, self.L
            )


    def normalize(self):
        """
        Normalization step.
        """
        return normalize_Phi_Z(self.Phi, self.Z)
    
    def CSC(self, Phi_perso):
        """
        One iteration of the CSC step.
        """
        return _CSC(self.X, Phi_perso, self.Z, S=self.S, penalty=self.penalty)

    def CDU(self):
        """
        One iteration of the CDU step.
        No personalization is considered.
        """
        return _CDU(self.X, self.Phi, self.Z, step_size=self.step_size, nb_steps=25)
    
    def CDU_perso(self):
        """
        One iteration of the CDU step.
        No personalization is considered.
        """
        return _CDU_perso(
            self.X, self.Phi, self.Z, self.A,
            step_size=self.step_size, nb_steps=25, 
            D=self.f.D, W=self.f.W, L=self.L
        )
    
    def IPU(self, Phi):
        """
        One iteration of the IPU step.
        """
        return _IPU(
            self.X, Phi, self.Z, self.A, 
            step_size=self.step_size, nb_steps=25,
            D=self.f.D, W=self.f.W, L=self.L
        )
    
    def plot_reconstruction(self):
        """
        Plot reconstruction.
        """

        # Define colors
        c_signal = 'lightslategrey'
        c_init = 'lightslategrey'
        c_common = 'cornflowerblue'
        c_perso = 'hotpink'

        f = plt.figure(layout='constrained', figsize=(9, 3))
        gs = GridSpec(self.S, 5, figure=f)

        axes_signals = [
            f.add_subplot(gs[i, :-1])
            for i in range(self.S)
        ]
        axes_atoms = [
            f.add_subplot(gs[i, -1])
            for i in range(self.S)
        ]

        def _plot_reconstruction(axes):
            
            # Create time vector
            t = np.arange(self.X.shape[-1])

            # Get personalized atoms
            D = self.f.personalize(self.Phi, self.A)

            # Plots
            for s, ax in enumerate(axes):

                # Input signal
                ax.plot(t, self.X[s, :], c=c_signal, lw=2, alpha=0.5, label='Input signal')

                for k in range(self.K):
                    once = True
                    
                    # Activations
                    alpha_max = abs(max(self.Z[s, k, :].min(), self.Z[s, k, :].max(), key=abs))
                    for j in np.arange(self.N-self.L):
                        z = self.Z[s, k, j]
                        if z != 0:
                            alpha = abs(z / alpha_max)
                            ax.axvline(j, alpha=alpha, lw=1.5, c=c_perso)

                            # Reconstruction
                            if abs(z) > 0.1:
                                ax.plot(t[j:j+self.L], z * D[k, s, :], c=c_perso, 
                                    label=f'Personalized atom {k}' if (s==len(axes)-1 and once) else ''
                                )
                                once = False

        def _plot_atoms(axes):

            # Get personalized atoms
            D_perso = self.f.personalize(self.Phi, self.A)

            for i, ax in enumerate(axes):

                # Horizontal line at 0
                ax.axhline(c='lightgrey', alpha=0.5, ls='--')

                for k in range(self.K):
                    
                    # Init
                    ax.plot(
                        self.Phi_init[k, :] / np.linalg.norm(self.Phi_init[k, :]), 
                        c=c_init, ls='--', alpha=0.8, label=f'Initialization'
                    )

                    # Common
                    ax.plot(
                        self.Phi[k, :] / np.linalg.norm(self.Phi[k, :]), 
                        c=c_common, alpha=0.8, label=f'Common atom'
                    )

                    # Perso
                    ax.plot(
                        D_perso[k, i, :] / np.linalg.norm(D_perso[k, i, :]), 
                        lw=1.3, c=c_perso, alpha=1, label=f'Perso. atom'
                    )

            # handles, labels = axes[1].get_legend_handles_labels()
            # order = [0, 1, 4, 2, 3, 5]
            
            axes[1].legend(
                # [handles[idx] for idx in order],[labels[idx] for idx in order],
                loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9
            )

        # Plot reconstruction
        _plot_reconstruction(axes_signals)

        # Plot atoms
        _plot_atoms(axes_atoms)

        # Customization
        for ax in (axes_signals + axes_atoms):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
        for ax in axes_signals[:-1]:
            ax.set_xticklabels([])
        for ax in axes_atoms[:-1]:
            ax.set_xticklabels([])
        # Do not show border effects
        for ax in axes_signals:
            ax.set_xlim((-1, self.N-1.2*self.L))

        plt.tight_layout()
        plt.show()
        plt.savefig(f'results.png', pad_inches=0, dpi=300)
        plt.close()