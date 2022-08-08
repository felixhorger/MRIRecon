import espirit_python.cfl as cfl
from espirit_python.espirit import espirit, espirit_proj, ifft
import h5py

# Load data
X = cfl.readcfl("espirit_python/data/knee")

# Derive ESPIRiT operator
esp = espirit(X, 6, 24, 0.01, 0.9925)
# Do projections
x = ifft(X, (0, 1, 2))
ip, proj, null = espirit_proj(x, esp)

# Save and quit
f = h5py.File("espirit_test.h5", "w")
f.create_dataset("data", data=X[0, :, :, :])
f.create_dataset("coil_sensitivities", data=esp[0, :, :, :, 0])
f.close()

