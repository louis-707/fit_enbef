import numpy as np
import scipy.sparse as sp


class mesh:
    def __init__(self, xmesh, ymesh, zmesh):
        self.x = xmesh
        self.y = ymesh
        self.z = zmesh
        self.Nx = len(xmesh)
        self.Ny = len(ymesh)
        self.Nz = len(zmesh)
        self.Np = self.Nx * self.Ny * self.Nz

    def __create_p(self, Mw):
        Np = self.Np
        main = -np.ones(Np)
        upper = np.ones(Np - Mw)

        P = sp.diags(  # func. diags baut dünnbesetze matrix aus diagonalen auf
            [main, upper],  # liste der diag vektoren
            [0, Mw],  # pos. der diag, 0= mitte, Mw = Mw abstand nach rechts
            shape=(Np, Np),
            # format='lil' #format lists of lits weil sich leichter modifzieren lässt, später dann return in 'csr' format
        )

        # ?? korrektur der Geisterkanten etc.

        return P.tocsr()

    def primal_px(self):
        # Mx=1 ...
        return self.__create_p(1)

    def primal_py(self):
        # My=Nx
        return self.__create_p(self.Nx)

    def primal_pz(self):
        # Ny*Nx
        return self.__create_p(self.Nx * self.Ny)

    def dual_div(self):
        Px = self.primal_px()
        Py = self.primal_py()
        Pz = self.primal_pz()

        return sp.hstack([-Px.T, -Py.T, -Pz.T], format="csr")
