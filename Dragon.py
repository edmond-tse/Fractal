import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

max_iter = 1000

N = 1000
init = torch.tensor([0 + 0j, 1 + 0j], device=device, dtype=torch.complex64)
# init = torch.tensor([0 + 0j, 1 + 0j, 1 - 1j], device=device, dtype=torch.complex64) # twindragon
z = init[torch.randint(0, 2, (N,), device=device)]
pts = torch.empty((max_iter, N), device=device, dtype=torch.complex64)
s1 = torch.tensor(1 + 1j, device=device, dtype=torch.complex64) / 2  # (1 + i) / 2
s2 = torch.tensor(1 - 1j, device=device, dtype=torch.complex64) / 2  # (1 - i) / 2
x = torch.tensor(1 + 0j, device=device, dtype=torch.complex64)  # 1


def f1(z):
    return s1 * z

def f2(z):
    return x - s2 * z

for i in range(max_iter):
    mask = torch.rand(N, device=device) < 0.5
    z = torch.where(mask, f1(z), f2(z))
    pts[i] = z

pts_np = pts.cpu().numpy()
X = pts_np.real.flatten()
Y = pts_np.imag.flatten()
fig = plt.figure(figsize=(16, 9))

plt.scatter(X, Y, color='blue', s=0.4)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
