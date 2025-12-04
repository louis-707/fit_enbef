import fit
from pyevtk.hl import gridToVTK
import numpy as np


Nx = Ny = Nz = 21
xmesh = np.linspace(-1, 1, Nx)
ymesh = np.linspace(-1, 1, Ny)
zmesh = np.linspace(-1, 1, Nz)

model = fit.Mesh(xmesh, ymesh, zmesh)

S_tilde = model.dual_div
G = -S_tilde.T


# X, Y, Z = np.meshgrid(xmesh, ymesh, zmesh, indexing="ij")
# Phi = X**2 * np.sin(2 * np.pi * Z)
# phi = Phi.reshape(model.Np)


phi = np.zeros(model.Np)
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            idx = model.canonical_index(i, j, k)
            x = xmesh[i]
            z = zmesh[k]
            phi[idx] = x**2 * np.sin(2 * np.pi * z)


ebow = -G.dot(phi)


exyz = (
    ebow[0 : model.Np],
    ebow[model.Np : 2 * model.Np],
    ebow[2 * model.Np : 3 * model.Np],
)
gridToVTK("./ex7", xmesh, ymesh, zmesh, pointData={"ebow": exyz})
