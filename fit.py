from functools import cached_property

import numpy as np
from numpy import ndarray, dtype, float64
from scipy import sparse as sp


class Mesh:
    """Discretized domain for the Finite Integration Technique (FIT).

    Parameters
    ----------
    xmesh : array_like
        x-coordinates of the grid points.
    ymesh : array_like
        y-coordinates of the grid points.
    zmesh : array_like
        z-coordinates of the grid points.

    Examples
    --------
    >>> Mesh(np.[0,1,2], np.[0,1,2], np.[0,1,2])
    """

    def __init__(self, xmesh: np.ndarray, ymesh: np.ndarray, zmesh: np.ndarray):
        """
        Args:
            xmesh (array_like): x-coordinates of the grid points.
            ymesh (array_like): y-coordinates of the grid points.
            zmesh (array_like): z-coordinates of the grid points.
        """
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.zmesh = zmesh

    @cached_property
    def Nx(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the x direction.
        """
        return len(self.xmesh)

    @cached_property
    def Ny(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the y direction."""
        return len(self.ymesh)

    @cached_property
    def Nz(self) -> int:
        """
        Returns
        -------
        int
            Number of grid points along the z direction.
        """
        return len(self.zmesh)

    @cached_property
    def Mx(self) -> int:
        """
        Returns
        -------
        int
            Step width in x direction.
        """
        return 1

    @cached_property
    def My(self) -> int:
        """
        Returns
        -------
        int
            Step width in y direction.
        """
        return self.Nx

    @cached_property
    def Mz(self) -> int:
        """
        Returns
        -------
        int
            Step width in z direction.
        """
        return self.Nx * self.Ny

    @cached_property
    def Np(self) -> int:
        """
        Returns
        -------
        int
            Total number of grid points.
        """
        return self.Nx * self.Ny * self.Nz

    def canonical_index(self, i: int, j: int, k: int) -> int:
        """Calculate and return the canonical index of a point given in xyz-indexing.

        Parameters
        ----------
        i : int
            Index in x-direction
        j : int
            Index in y-direction
        k : int
            Index in z-direction

        Returns
        -------
        int
            Canonical index of the given grid point
        """
        return (k - 1) * self.Mz + (j - 1) * self.My + i

    def canonical_inv(self, n: int) -> tuple[int, int, int]:
        """Calculate the xyz-indexing of a point given its canonical index.

        Parameters
        ----------
        n : int
            Canonical index of point

        Returns
        -------
        tuple[int, int, int]
            xyz-indexing of point as a tuple
        """
        n = n - 1
        i = n % self.Nx + 1
        n = (n - i + 1) // self.Nx
        j = n % self.Ny + 1
        n = (n - j + 1) // self.Ny
        k = n + 1
        return i, j, k

    @cached_property
    def primal_idxs(self) -> ndarray[tuple[int], dtype[bool]]:
        """Return boolean array containing the value false if corresponding edge is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise (sorted according to global canonical indexing)
        """
        raise ("Implement in 7.3")
        idxs = None

        return idxs

    @cached_property
    def primal_idxa(self) -> ndarray[tuple[int], dtype[float64]]:
        """Return boolean array containing the value false if corresponding face is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise (sorted according to global canonical indexing)
        """
        raise ("Implement in 7.3")
        primal_idxa = None

        return primal_idxa

    @cached_property
    def primal_idxv(self) -> ndarray[tuple[int], dtype[bool]]:
        """Return boolean array containing the value false if corresponding cell is a ghost-element and true otherwise.

        Returns
        -------
        np.ndarray (bool)
            False if ghost-element, true otherwise.
        """
        raise ("Implement in 7.3")
        primal_idxv = None

        return primal_idxv

    def __create_p(self, offset: int) -> sp.csr_array:
        """Return sparse matrix with value -1 on main diagonal and 1 on second diagonal shifted by
        offset in relation to the first one. All other elements are 0.

        Parameters
        ----------
        offset : int
            distance between diagonals

        Returns
        -------
        sp.csr.csr_array
            matrix with value -1 on main diagonal, 1 on diagonal shifted by offset and 0 else

        """
        Np = self.Np
        main = -np.ones(Np)
        upper = np.ones(Np - offset)

        P = sp.diags(  # func. diags baut dünnbesetze matrix aus diagonalen auf
            [main, upper],  # liste der diag vektoren
            [0, offset],  # pos. der diag, 0= mitte, Mw = Mw abstand nach rechts
            shape=(Np, Np),
            format="lil",  # format lists of lits weil sich leichter modifzieren lässt, später dann return in 'csr' format
        )

        # ?? korrektur der Geisterkanten etc.

        return P.tocsr()

    @cached_property
    def primal_px(self) -> sp.csr_array:
        """
        Returns
        -------
        sp.csr_array
            Matrix representing partial derivative in respect to x.
        """

        return self.__create_p(1)

    @cached_property
    def primal_py(self) -> sp.csr_array:
        """
        Returns
        -------
        sp.csr_array
            Matrix representing partial derivative in respect to y.
        """
        return self.__create_p(self.Nx)

    @cached_property
    def primal_pz(self) -> sp.csr_array:
        """
        Returns
        -------
        sp.csr_array
            Matrix representing partial derivative in respect to z.
        """
        return self.__create_p(self.Nx * self.Ny)

    @cached_property
    def primal_grad(self) -> sp.csr_array:
        """
        Returns
        -------
        sp.csr_array
            Matrix representing discrete gradient operator for primal grid.
        """
        return -self.dual_div.transpose()

    @cached_property
    def dual_div(self) -> sp.csr_array:
        """
        Returns
        -------
        sp.csr_array
            Matrix representing discrete divergence operator for dual grid.
        """

        Px = self.primal_px
        Py = self.primal_py
        Pz = self.primal_pz

        return sp.hstack([-Px.T, -Py.T, -Pz.T], format="csr")

    @cached_property
    def dual_idxs(self) -> ndarray[tuple[int], dtype[bool]]:
        """Return boolean array with value false if corresponding edge is no dual edge of the grid.

        Returns
        -------
        ndarray (boolean)
            false if element outside primal grid, true else
        """
        raise ("Implement in 7.3")
        return None

    @cached_property
    def dual_idxa(self) -> ndarray[tuple[int], dtype[bool]]:
        """Return boolean array with value false if corresponding face is no dual face of the grid.

        Returns
        -------
        ndarray (boolean)
            False if element outside primal grid, true else
        """
        raise ("Implement in 7.3")
        return None

    def null_inv(self, A: sp.csr_array) -> sp.csr_array:
        """Calculate pseudo inverse of given sparse diagonal matrix by inversing non-zero diagonal elements."""
        diagonal = A.diagonal()
        inverted_diagonal = np.divide(
            1, diagonal, where=diagonal != 0, out=np.zeros_like(diagonal)
        )
        return sp.diags_array(inverted_diagonal, format="csr")
