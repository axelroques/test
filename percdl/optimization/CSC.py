
import csc


def _CSC(X, Phi_perso, Z, S, penalty):

    for s in range(S):
        Z[s, :, :] = csc.update_z(
            [X[s, :]],
            dictionary=Phi_perso.transpose(1, 0, 2)[s, :, :], 
            penalty=penalty,
            constraint_str=csc.NO_CONSTRAINT
        )[0].T

    return Z