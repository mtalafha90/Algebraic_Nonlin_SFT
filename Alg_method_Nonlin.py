import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import numpy.ma as ma

# ------------------------
# CONFIGURATION
# ------------------------
np.random.seed(42)

NUM_CYCLES = 25
NUM_ARS = 200
A0 = 0.015
AMP_RATIOS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
lambda_R_values = np.linspace(8.0, 20.0, 30)
b_joy = 0.15
b_lat = 2.4

lambda_base = 17
sigma_lambda = 6
flux_mu, flux_sigma = 22.3, 0.3

# ------------------------
# FUNCTIONS
# ------------------------
def emergence_profile(t, a=0.00185, b=18.0, c=0.71, t0=0):
    b = 27.12+25.15/(a*10**3)**(1/4)
    numer = a * (t - t0)**3
    denom = np.exp(((t - t0)**2) / b**2) - c
    return np.where(denom > 0, numer / denom, 0.0)

def apply_tilt_quenching(delta_lambda_0, amp_ratio):
    return delta_lambda_0 * (1 - b_joy * (amp_ratio - 1))

def apply_latitude_quenching(lambda_0, amp_ratio):
    lam_q =  lambda_0 + b_lat * (amp_ratio - 1)
    return np.clip(lam_q, 5.0, 35.0)  # Prevents unrealistically low or negative latitudes
'''
def generate_ar(lat, flux, tilt_deg, asym=True):
    tilt_rad = np.radians(tilt_deg)
    separation = 5
    delta_lat = (separation / 2.0) * np.sin(tilt_rad)

    if asym:
        imbalance = 1.0 # Flux ratio (0.7 to 1.3)
        flux_pos = flux * imbalance/(1 + imbalance)
        flux_neg = flux - flux_pos
    else:
        flux_pos = flux / 2
        flux_neg = -flux / 2

    pos_polarity = (lat + delta_lat, flux_pos)
    neg_polarity = (lat - delta_lat, -flux_neg)

    return [pos_polarity, neg_polarity]
'''

def compute_dipole_contribution(polarities, lambda_R):
    dipole = 0.0
    for lat, flux in polarities:
        erf_arg = abs(lat) / (np.sqrt(2) * lambda_R)
        erf_val = erf(erf_arg)
        suppression = erf_val# * (abs(lat) / 90.0)  # Additional weight to penalize near-equator
        dipole += flux * suppression * np.sign(lat)
    return dipole
'''
def generate_ar(lat, flux, tilt_deg, asym=False):
    tilt_rad = np.radians(tilt_deg)
    base_separation = 5  # Total angular separation

    # Default: equal displacement
    delta_lat_lead = (base_separation / 2.0) * np.sin(tilt_rad)
    delta_lat_follow = -delta_lat_lead

    if asym:
        # Morphological asymmetry: disperse following polarity more
        spread_factor = 2.0  # e.g., following polarity is 2× more dispersed
        delta_lat_lead = (base_separation / (1 + spread_factor)) * np.sin(tilt_rad)
        delta_lat_follow = -spread_factor * delta_lat_lead

    # Fluxes are equal and opposite
    flux_pos = flux / 2
    flux_neg = -flux / 2

    pos_polarity = (lat + delta_lat_lead, flux_pos)
    neg_polarity = (lat + delta_lat_follow, flux_neg)

    return [pos_polarity, neg_polarity]
'''
def generate_ar(lat, flux, tilt_deg, tilt_asym_deg=0.5, use_tilt_asym=False, use_morph_asym=True):
    """
    Generate a bipolar active region with optional tilt and morphological asymmetries.

    Parameters:
    - lat: Latitude of the AR
    - flux: Total unsigned flux
    - tilt_deg: Nominal tilt angle (in degrees)
    - tilt_asym_deg: Asymmetry added/subtracted from tilt angle (in degrees)
    - use_tilt_asym: Apply asymmetry in tilt angle (bool)
    - use_morph_asym: Apply morphological spread asymmetry (bool)

    Returns:
    - List of two tuples: [(lat_pos, +flux/2), (lat_neg, -flux/2)]
    """
    tilt_rad = np.radians(tilt_deg)
    base_separation = 5.0  # Total angular separation in degrees

    if use_tilt_asym:
        delta = np.radians(tilt_asym_deg)
        # Apply asymmetric tilt: one polarity slightly more/less tilted
        delta_lat_lead = (base_separation / 2.0) * np.sin(tilt_rad + delta)
        delta_lat_follow = -(base_separation / 2.0) * np.sin(tilt_rad - delta)
    else:
        # Symmetric tilt
        delta_lat_lead = (base_separation / 2.0) * np.sin(tilt_rad)
        delta_lat_follow = -delta_lat_lead

    if use_morph_asym:
        # Morphological asymmetry: spread the following polarity more
        spread_factor = 2.0
        #spread_factor = np.random.uniform(1.5, 3.0)
        delta_lat_lead = (base_separation / (1 + spread_factor)) * np.sin(tilt_rad)
        delta_lat_follow = -spread_factor * delta_lat_lead

    flux_pos = flux / 2
    flux_neg = -flux / 2

    pos_polarity = (lat + delta_lat_lead, flux_pos)
    neg_polarity = (lat + delta_lat_follow, flux_neg)

    return [pos_polarity, neg_polarity]


# ------------------------
# MAIN SIMULATION
# ------------------------

# Cycle length and time grid (months)
t_months = np.linspace(0, 132, 500)
profile = emergence_profile(t_months)
profile = np.clip(profile, 0, None)
pdf = profile / np.trapezoid(profile, t_months)
cdf = np.cumsum(pdf)
cdf /= cdf[-1]
tilt_asym_deg = np.random.normal(loc=0.0, scale=1.0)

# Sample emergence times for one cycle
rng = np.random.default_rng(seed=42)
sampled_times_all = np.interp(rng.uniform(0, 1, NUM_ARS * NUM_CYCLES), cdf, t_months)


final_dipoles_grid = {
    "noQ": np.zeros((len(AMP_RATIOS), len(lambda_R_values))),
    "TQ": np.zeros((len(AMP_RATIOS), len(lambda_R_values))),
    "LQ": np.zeros((len(AMP_RATIOS), len(lambda_R_values))),
    "LQTQ": np.zeros((len(AMP_RATIOS), len(lambda_R_values)))
}

for i, amp_ratio in enumerate(AMP_RATIOS):
    for j, lambda_R in enumerate(lambda_R_values):
        dipoles = {mode: [] for mode in final_dipoles_grid}
        for cyc in range(NUM_CYCLES):
            D_total = {mode: 0.0 for mode in final_dipoles_grid}
            sampled_times = sampled_times_all[cyc * NUM_ARS : (cyc + 1) * NUM_ARS]
            for ar_idx in range(NUM_ARS):
                t = sampled_times[ar_idx]
                # Linearly decrease emergence latitude over time (mimicking butterfly)
                lambda_max, lambda_min = 35, 5
                lambda_0 = lambda_max - (lambda_max - lambda_min) * (t / max(t_months))
                scatter_std = 7 - 4 * (t / max(t_months))  # From 7 to 3 degrees
                lambda_0 += np.random.normal(0, scatter_std)
                #lambda_0 += np.random.normal(0, sigma_lambda)  # optional scatter
                lambda_0 = np.clip(lambda_0, 3.0, 40.0)
                log_flux = np.random.normal(flux_mu, flux_sigma)
                rel_flux = 10**log_flux
                flux =  A0 * amp_ratio * rel_flux

                tilt = 1.5 * np.sin(np.radians(lambda_0))

                # noQ
                lam = lambda_0
                tilt = 1.5 * np.sin(np.radians(lam))
                polarities = generate_ar(lam, flux, tilt, tilt_asym_deg=tilt_asym_deg, use_tilt_asym=False, use_morph_asym=True)
                D_total["noQ"] += compute_dipole_contribution(polarities, lambda_R)

                # TQ
                tilt_TQ = apply_tilt_quenching(tilt, amp_ratio)
                polarities_TQ = generate_ar(lam, flux, tilt_TQ, tilt_asym_deg=tilt_asym_deg, use_tilt_asym=False, use_morph_asym=True)
                D_total["TQ"] += compute_dipole_contribution(polarities_TQ, lambda_R)

                # LQ
                lam_LQ = apply_latitude_quenching(lambda_0, amp_ratio)
                lam_LQ = np.clip(lam_LQ, 3.0, 35.0)
                tilt_LQ = 1.5 * np.sin(np.radians(lam_LQ))
                polarities_LQ = generate_ar(lam_LQ, flux, tilt_LQ, tilt_asym_deg=tilt_asym_deg, use_tilt_asym=False, use_morph_asym=True)
                D_total["LQ"] += compute_dipole_contribution(polarities_LQ, lambda_R)

                # LQTQ
                lam_LQTQ = apply_latitude_quenching(lambda_0, amp_ratio)
                lam_LQTQ = np.clip(lam_LQTQ, 3.0, 35.0)
                tilt_LQTQ = 1.5 * np.sin(np.radians(lam_LQTQ))
                tilt_LQTQ = apply_tilt_quenching(tilt_LQTQ, amp_ratio)
                polarities_LQTQ = generate_ar(lam_LQTQ, flux, tilt_LQTQ, tilt_asym_deg=tilt_asym_deg, use_tilt_asym=False, use_morph_asym=True)
                D_total["LQTQ"] += compute_dipole_contribution(polarities_LQTQ, lambda_R)

            for mode in final_dipoles_grid:
                dipoles[mode].append(D_total[mode])
        
        for mode in final_dipoles_grid:
            final_dipoles_grid[mode][i, j] = np.mean(dipoles[mode]) / 1e22

# ------------------------
# SUPPRESSION DIFFERENCES
# ------------------------

delta_TQ = final_dipoles_grid["noQ"] - final_dipoles_grid["TQ"]
delta_LQ = final_dipoles_grid["noQ"] - final_dipoles_grid["LQ"]
delta_LQTQ = final_dipoles_grid["noQ"] - final_dipoles_grid["LQTQ"]

# ------------------------
# HEATMAPS
# ------------------------

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

modes = ["noQ", "TQ", "LQ", "LQTQ"]
titles = ["No Quenching", "Tilt Quenching", "Latitude Quenching", "Tilt+Latitude Quenching"]
X, Y = np.meshgrid(lambda_R_values, AMP_RATIOS)

for idx, mode in enumerate(modes):
    row, col = divmod(idx, 2)
    ax = axs[row, col]
    c = ax.pcolormesh(X, Y, final_dipoles_grid[mode], shading='auto', cmap='viridis')
    ax.set_title(titles[idx])
    ax.set_ylabel("Cycle Amplitude (A_n / A_0)")
    if row == 1:
        ax.set_xlabel("$\\lambda_{R}[ ^{\\circ}]$")
    fig.colorbar(c, ax=ax, label="Dipole ($\\times 10^{22}$)")

plt.suptitle("Final Dipole Moments")
plt.tight_layout()
plt.show()


'''
#-------------------------
# 3D SURFACE PLOT
#-------------------------

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, delta_LQTQ, cmap='plasma')
ax.set_title('Suppression Surface (NoQ - LQTQ)')
ax.set_xlabel("λ_R (deg)")
ax.set_ylabel("Cycle Amplitude (A_n / A_0)")
ax.set_zlabel("Suppression ($\times 10^{22}$)")
plt.show()

'''
# ------------------------
# RATIO PLOT (LQ vs TQ)
# ------------------------

ratio_LQ_TQ = np.ma.divide(delta_LQ, delta_TQ)
fig2, ax = plt.subplots(figsize=(8, 6))
c = ax.pcolormesh(X, Y, ratio_LQ_TQ, shading='auto', cmap='coolwarm')
# Add contours on top of the heatmap
contour_levels = np.linspace(np.min(ratio_LQ_TQ), np.max(ratio_LQ_TQ), 10)
CS = ax.contour(X, Y, ratio_LQ_TQ, levels=contour_levels, colors='k', linewidths=0.8)
ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")  # Label contours
ax.set_title("Relative importance of LQ/TQ")
ax.set_xlabel("$\\lambda_{R} [^{\\circ}]$")
ax.set_ylabel("Cycle Amplitude ($A_{n} / A_{0}$)")
fig2.colorbar(c, ax=ax, label="$dev_{LQ}/dev_{TQ}$")
plt.tight_layout()
plt.show()


# ------------------------
# DIPOLE vs AMPLITUDE for SELECTED λ_R
# ------------------------
lambda_R_index = 12
lambda_R_single = lambda_R_values[lambda_R_index]

noQ_vals = final_dipoles_grid["noQ"][:, lambda_R_index]
TQ_vals  = final_dipoles_grid["TQ"][:, lambda_R_index]
LQ_vals  = final_dipoles_grid["LQ"][:, lambda_R_index]
LQTQ_vals = final_dipoles_grid["LQTQ"][:, lambda_R_index]

plt.figure(figsize=(10, 6))
plt.plot(AMP_RATIOS, noQ_vals, marker='o', label='No Quenching', color='black')
plt.plot(AMP_RATIOS, TQ_vals, marker='s', label='Tilt Quenching', color='red')
plt.plot(AMP_RATIOS, LQ_vals, marker='^', label='Latitude Quenching', color='blue')
plt.plot(AMP_RATIOS, LQTQ_vals, marker='D', label='Tilt + Latitude Quenching', color='green')

plt.xlabel("Cycle Amplitude (A_n / A_0)")
plt.ylabel("Final Dipole Moment ($\\times 10^{22}$)")
plt.title("Dipole vs Cycle Amplitude $\\lambda_{R}$ ="+f" {lambda_R_single:.2f}"+"$ [^{\\circ}]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
'''
import matplotlib.animation as animation

fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
line_noQ, = ax_anim.plot([], [], 'ko-', label='No Quenching')
line_TQ,  = ax_anim.plot([], [], 'rs-', label='Tilt Quenching')
line_LQ,  = ax_anim.plot([], [], 'b^-', label='Latitude Quenching')
line_LQTQ, = ax_anim.plot([], [], 'gd-', label='Tilt + Latitude Quenching')
title = ax_anim.text(0.5, 1.05, '', transform=ax_anim.transAxes, ha='center', fontsize=14)

ax_anim.set_xlim(min(AMP_RATIOS), max(AMP_RATIOS))
ax_anim.set_ylim(0, np.max(final_dipoles_grid["noQ"]) * 1.1)
ax_anim.set_xlabel("Cycle Amplitude (A_n / A_0)")
ax_anim.set_ylabel("Final Dipole Moment ($\times 10^{22}$)")
ax_anim.grid(True)
ax_anim.legend()

def update(frame):
    lambda_R_current = lambda_R_values[frame]
    title.set_text(f"Dipole vs Cycle Amplitude @ λ_R = {lambda_R_current:.2f}°")

    line_noQ.set_data(AMP_RATIOS, final_dipoles_grid["noQ"][:, frame])
    line_TQ.set_data(AMP_RATIOS, final_dipoles_grid["TQ"][:, frame])
    line_LQ.set_data(AMP_RATIOS, final_dipoles_grid["LQ"][:, frame])
    line_LQTQ.set_data(AMP_RATIOS, final_dipoles_grid["LQTQ"][:, frame])
    return line_noQ, line_TQ, line_LQ, line_LQTQ, title

ani = animation.FuncAnimation(
    fig_anim, update, frames=len(lambda_R_values), interval=400, blit=True, repeat=True
)

plt.tight_layout()
plt.show()
'''

# ------------------------
# DEVIATIONS FROM NoQ AT amp_ratio = 3.0
# ------------------------

amp_target = 3.0
amp_index = AMP_RATIOS.index(amp_target)
deviations = {"TQ": [], "LQ": [], "LQTQ": []}

fig_all, axs_all = plt.subplots(6, 5, figsize=(24, 20), sharex=True, sharey=True)
axs_all = axs_all.flatten()

for idx, lambda_R_val in enumerate(lambda_R_values):
    ax = axs_all[idx]
    
    noQ_vals = final_dipoles_grid["noQ"][:, idx]
    TQ_vals  = final_dipoles_grid["TQ"][:, idx]
    LQ_vals  = final_dipoles_grid["LQ"][:, idx]
    LQTQ_vals = final_dipoles_grid["LQTQ"][:, idx]

    ax.plot(AMP_RATIOS, noQ_vals, 'ko-', label='NoQ')
    ax.plot(AMP_RATIOS, TQ_vals, 'rs--', label='TQ')
    ax.plot(AMP_RATIOS, LQ_vals, 'b^--', label='LQ')
    ax.plot(AMP_RATIOS, LQTQ_vals, 'gd--', label='LQTQ')
    ax.set_title("$\\lambda_{R}$ ="+f" {lambda_R_val:.2f}"+"$ [^{\\circ}]$")
    ax.grid(True)

    # Compute deviations at amp_ratio = 3.0
    dev_TQ = noQ_vals[amp_index] - TQ_vals[amp_index]
    dev_LQ = noQ_vals[amp_index] - LQ_vals[amp_index]
    dev_LQTQ = noQ_vals[amp_index] - LQTQ_vals[amp_index]

    deviations["TQ"].append(dev_TQ)
    deviations["LQ"].append(dev_LQ)
    deviations["LQTQ"].append(dev_LQTQ)

# Final plot adjustments
for ax in axs_all:
    ax.label_outer()
    ax.set_ylim(0, np.max(final_dipoles_grid["noQ"]) * 1.1)

axs_all[0].set_ylabel("Dipole Moment ($\\times 10^{22}$)")
axs_all[-5].legend(loc='upper right', fontsize='small')
fig_all.suptitle(f"Dipole Moment vs Cycle Amplitude — Deviation Computed at A_n/A_0 = {amp_target}", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# ------------------------
# Plot deviations vs λ_R
# ------------------------

plt.figure(figsize=(10, 6))
plt.plot(lambda_R_values, deviations["TQ"], 'rs-', label='Δ(NoQ - TQ)')
plt.plot(lambda_R_values, deviations["LQ"], 'b^-', label='Δ(NoQ - LQ)')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("$\\lambda_{R} [^{\\circ}]$")
plt.ylabel("Deviations from NoQ ($\\times 10^{22}$)")
plt.title("Deviations at A_n/A_0 = 3.0")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''

plt.figure(figsize=(10, 6))
#plt.plot(lambda_R_values, deviations["TQ"], 'rs-', label='Δ(NoQ - TQ)')
plt.plot(lambda_R_values, np.array(deviations["LQ"])/np.array(deviations["TQ"]), 'b^-', label='Δ(NoQ - LQ)/Δ(NoQ - TQ)')
#plt.plot(lambda_R_values, deviations["LQTQ"], 'gd-', label='Δ(NoQ - LQTQ)')
#plt.axhline(0, color='gray', linestyle='--')
#plt.ylim(0.8,0.9)
plt.xlabel("λ_R (deg)")
plt.ylabel("Deviation from NoQ ($\\times 10^{22}$)")
plt.title(f"Dipole Moment Suppression at Cycle Amplitude = {amp_target}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

from scipy.optimize import curve_fit

# Define the model function
def suppression_model(lambda_r, c1, c2):
    return c1 + c2 / (lambda_r ** 2)

# Prepare data
x_data = lambda_R_values
y_data = np.array(deviations["LQ"]) / np.array(deviations["TQ"])

# Fit the model
popt, pcov = curve_fit(suppression_model, x_data, y_data)
c1_fit, c2_fit = popt

# Generate fit curve
lambda_r_fit = np.linspace(min(x_data), max(x_data), 300)
fit_vals = suppression_model(lambda_r_fit, *popt)

# Plot the original data and the fit
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'b^-', label='$dev_{LQ}/dev_{TQ}$')
plt.plot(lambda_r_fit, fit_vals, 'k--', label='Fit: c1 + c2/$\\lambda_{R}^{2}$\n'f'c1={c1_fit:.4f}, c2={c2_fit:.2f}')
plt.xlabel("$\\lambda_{R} [^{\\circ}]$")
plt.ylabel("$dev_{LQ}/dev_{TQ}$")
plt.title(f"Fit at Cycle Amplitude = {amp_target}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ------------------------
# BUTTERFLY DIAGRAM VISUALIZATION
# ------------------------
latitudes = []
for t in sampled_times_all:
    lam_base = lambda_max - (lambda_max - lambda_min) * (t / max(t_months))
    scatter_std = 7 - 4 * (t / max(t_months))
    lam_with_scatter = lam_base + np.random.normal(0, scatter_std)
    latitudes.append(np.clip(lam_with_scatter, 3, 40))

plt.figure(figsize=(10, 4))
plt.scatter(sampled_times_all, latitudes, s=2, alpha=0.5)
plt.xlabel("Time [months]")
plt.ylabel("Emergence Latitude [deg]")
plt.title("Synthetic Butterfly Diagram with Time-Dependent Latitude and Scatter")
plt.grid(True)
plt.tight_layout()
plt.show()
