import fit
from pyevtk.hl import gridToVTK
import numpy as np


Nx = Ny = Nz = 100
xmesh = np.linspace(-2, 2, Nx)
ymesh = np.linspace(-2, 2, Ny)
zmesh = np.linspace(-2, 2, Nz)

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
            y = ymesh[j]
            z = zmesh[k]
            phi[idx] = +8 * np.exp(
                -(((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) / 0.18)
            ) - 8 * np.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.18))


ebow = -G.dot(phi)


exyz = (
    ebow[0 : model.Np],
    ebow[model.Np : 2 * model.Np],
    ebow[2 * model.Np : 3 * model.Np],
)
gridToVTK("./ex7", xmesh, ymesh, zmesh, pointData={"ebow": exyz, "phi": phi})


phi_111 = phi[model.canonical_index(1, 1, 1) - 1]
phi_211 = phi[model.canonical_index(2, 1, 1) - 1]
phi_112 = phi[model.canonical_index(1, 1, 2) - 1]
print(phi_111)
print(phi_211)
print(phi_112)

e1_manual = -(phi_211 - phi_111)

print("e_bow_1 =", e1_manual)
print("operator e1       =", ebow[0 : model.Np][0])


e2Np1_manual = -(phi_112 - phi_111)

print("ebow_(2Np+1) =", e2Np1_manual)
print("operator e_(2Np+1)      =", ebow[2 * model.Np : 3 * model.Np][0])
