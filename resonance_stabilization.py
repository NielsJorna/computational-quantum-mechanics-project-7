import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

m = 1
w = 1
hbar = 1

N = 200

barrier_start = 2
barrier_end = 8

L_min = 10
L_max = 50
L_steps = 1200

L_arr = np.linspace(L_min, L_max, L_steps)

n = np.arange(1, N + 1)
T_num = (hbar**2 * n**2 * np.pi**2) / (2 * m)


def V_matrix_well(L, N, height, R1, R2):
    n = np.arange(1, N + 1)
    i, j = np.meshgrid(n, n, indexing='ij')

    def V_integral(r):
        term_1 = np.sinc((i-j)*r/L) * (r/L)
        term_2 = np.sinc((i+j)*r/L) * (r/L)
        return term_1 - term_2

    V_mat = height * (V_integral(R2) - V_integral(R1))
    return V_mat


def V_matrix_sphere(L, N, A):
    r_grid = np.linspace(0, L, 2000)
    dr = r_grid[1] - r_grid[0]
    V_r = A * (r_grid**2) * np.exp(-r_grid)
    n_indices = np.arange(1, N + 1).reshape(-1, 1)
    phi = np.sqrt(2/L) * np.sin(n_indices * np.pi * r_grid / L)
    V_mat = (phi * V_r) @ phi.T * dr
    return V_mat


def get_dipole(L, N, ground_state_evec):
    r_grid = np.linspace(0, L, 2000)
    dr = r_grid[1] - r_grid[0]

    n_vec = np.arange(1, N + 1).reshape(-1, 1)
    phi = np.sqrt(2/L) * np.sin(n_vec * np.pi * r_grid / L)

    psi0 = ground_state_evec @ phi

    M = np.sum(phi * (psi0 * r_grid), axis=1) * dr
    return M


eigs_arr = []
dipoles_arr = []
ground_evecs_arr = []

A = 7.5

for L in tqdm(L_arr):
    T_mat = np.diag(T_num / L**2)
    V_mat = V_matrix_sphere(L, N, A)
    H = T_mat + V_mat

    energies, evecs = sp.linalg.eigh(H, eigvals_only=False, subset_by_index=[0, 49])

    ground_state_evec = evecs[:, 0]
    M_vector = get_dipole(L, N, ground_state_evec)

    dipole_moments = evecs.T @ M_vector
    dipoles_arr.append(dipole_moments)
    eigs_arr.append(energies)

eigs_arr = np.array(eigs_arr)

plt.figure(figsize=(10, 6))
plt.plot(L_arr, eigs_arr, 'k-', lw=0.5, alpha=0.6)

Er_approx = 3.426
plt.axhline(Er_approx, color='r', lw=1, linestyle='--', label=f'Literature $E_r$ = {Er_approx}')
plt.xlabel("Box Size $L$")
plt.ylabel("Energy $E$")
plt.title("Stabilization Graph")
plt.ylim(0, 10)
plt.legend()
plt.savefig("stabilization_graph.png")
plt.show()


slopes = np.gradient(eigs_arr, L_arr, axis=0)
flattened_energies = eigs_arr.flatten()
flattened_slopes = slopes.flatten()
flattened_dipoles = np.array(dipoles_arr).flatten()

epsilon = 1e-4

energy_bins = np.linspace(0, 10, 2400)

rho_plain, bin_edges = np.histogram(
    flattened_energies, bins=energy_bins,
    weights=1.0 / ((L_max - L_min) * (np.abs(flattened_slopes) + epsilon)),
    density=False
)

# threshold = 0.01  # a.u. per a.u. — tune naar wens
# mask = np.abs(flattened_slopes) > threshold
# rho_plain, bin_edges = np.histogram(
#     flattened_energies[mask], bins=energy_bins,
#     weights=1.0 / ((L_max - L_min) * np.abs(flattened_slopes[mask])),
# )

rho_fano, _ = np.histogram(
    flattened_energies, bins=energy_bins,
    weights=np.abs(flattened_dipoles)**2 / ((L_max - L_min) * (np.abs(flattened_slopes) + epsilon)),
    density=False
)

e_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

mask = e_centers > 2
rho_plain = rho_plain[mask]
rho_fano = rho_fano[mask]
e_centers = e_centers[mask]

from scipy.optimize import curve_fit


def resonance_fit(E, a, b, Er, gamma):
    bg = a * E + b
    lorentz = (1/np.pi) * (gamma/2) / ((E - Er)**2 + (gamma/2)**2)
    return bg + lorentz


def fano_fit(E, a, b, Er, q, gamma, A_fano):
    bg = a * E + b
    eps = (E - Er) / (gamma / 2)
    fano = A_fano * (q + eps)**2 / (1 + eps**2)
    return bg + fano


Er_lit, Gamma_lit = 3.426, 0.0255

initials = [0.001, 0.01, 3.45, 0.0255]
popt, pcov = curve_fit(
    resonance_fit, e_centers, rho_plain, p0=initials,
    bounds=([-np.inf, -np.inf, 3.0, 1e-6], [np.inf, np.inf, 4.0, 1.0])
)
perr = np.sqrt(np.diag(pcov))

plot_window = 0.3
E_plot = np.linspace(popt[2] - plot_window, popt[2] + plot_window, 500)

plt.figure(figsize=(8, 5))
plt.plot(e_centers, rho_plain, 'bo', ms=2, label='DOS data')
plt.plot(E_plot, resonance_fit(E_plot, *popt), 'r-', lw=1.5, label='Lorentzian fit')
plt.xlabel("Energy $E$")
plt.ylabel("Density of States $\\rho(E)$")
plt.title("Plain DOS — Lorentzian Fit")
plt.legend()
plt.savefig("lorentzian_fit.png")
plt.show()

print(f"\nLorentzian Fit Results")
print(f"{'':20} {'This fit':>12} {'±1σ':>10} {'Literature':>12}")
print(f"{'Er (a.u.)':20} {popt[2]:>12.4f} {perr[2]:>10.4f} {Er_lit:>12.4f}")
print(f"{'Γ (a.u.)':20} {popt[3]:>12.4f} {perr[3]:>10.4f} {Gamma_lit:>12.4f}")
print(f"{'τ = ħ/Γ (a.u.)':20} {hbar/popt[3]:>12.4f} {'':>10} {hbar/Gamma_lit:>12.4f}")

initials_fano = [0.1, 1, 3.45, 10, 0.03, 1.0]
popt_fano, pcov_fano = curve_fit(
    fano_fit, e_centers, rho_fano, p0=initials_fano,
    bounds=([-np.inf, -np.inf, 3.0, -np.inf, 1e-6, 0], [np.inf, np.inf, 4.0, np.inf, 1.0, np.inf])
)
perr_fano = np.sqrt(np.diag(pcov_fano))

E_plot_fano = np.linspace(popt_fano[2] - plot_window, popt_fano[2] + plot_window, 500)

plt.figure(figsize=(8, 5))
plt.plot(e_centers, rho_fano, 'go', ms=2, label='Dipole-weighted DOS')
plt.plot(E_plot_fano, fano_fit(E_plot_fano, *popt_fano), 'r-', lw=1.5, label='Fano fit')
plt.xlabel("Energy $E$")
plt.ylabel("$\\sigma(E)$ (arb. units)")
plt.title("Dipole-weighted DOS — Fano Fit")
plt.legend()
plt.savefig("fano_fit.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(e_centers, rho_plain, 'bo', ms=2, label='DOS data')
ax1.plot(E_plot, resonance_fit(E_plot, *popt), 'r-', lw=1.5, label='Lorentzian fit')
ax1.set_xlim(popt[2] - plot_window, popt[2] + plot_window)
ax1.set_xlabel("Energy $E$")
ax1.set_ylabel("$\\rho(E)$")
ax1.set_title("Plain DOS")
ax1.legend()

ax2.plot(e_centers, rho_fano, 'go', ms=2, label='Dipole-weighted DOS')
ax2.plot(E_plot_fano, fano_fit(E_plot_fano, *popt_fano), 'r-', lw=1.5, label='Fano fit')
ax2.set_xlim(popt_fano[2] - plot_window, popt_fano[2] + plot_window)
ax2.set_xlabel("Energy $E$")
ax2.set_ylabel("$\\sigma(E)$")
ax2.set_title("Dipole-weighted DOS (Fano)")
ax2.legend()

plt.suptitle("Lorentzian vs Fano Lineshape Comparison")
plt.tight_layout()
plt.show()

print(f"\nFano Fit Results")
print(f"{'':20} {'This fit':>12} {'±1σ':>10} {'Literature':>12}")
print(f"{'Er (a.u.)':20} {popt_fano[2]:>12.4f} {perr_fano[2]:>10.4f} {Er_lit:>12.4f}")
print(f"{'Γ (a.u.)':20} {popt_fano[4]:>12.4f} {perr_fano[4]:>10.4f} {Gamma_lit:>12.4f}")
print(f"{'τ = ħ/Γ (a.u.)':20} {hbar/popt_fano[4]:>12.4f} {'':>10} {hbar/Gamma_lit:>12.4f}")
print(f"{'q (asymmetry)':20} {popt_fano[3]:>12.4f} {perr_fano[3]:>10.4f} {'N/A':>12}")
