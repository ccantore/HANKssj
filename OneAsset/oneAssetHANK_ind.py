#1 asset HANK
import os
import pickle
mainpath='/Users/cristiano/Dropbox/My Mac (Cristiano’s MacBook Pro)/Documents/GitHub/HANKssj/OneAsset'

os.chdir(mainpath)

import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import simple, create_model  # functions
from sequence_jacobian import hetblocks, grids      # modules



hh = hetblocks.hh_labor.hh

print(hh)
print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')
print(f'Micro outputs: {hh.internals}')


def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

def wages(w, e_grid):
    we = w * e_grid
    return we

hh1 = hh.add_hetinputs([make_grid, transfers, wages])

print(hh1)
print(f'Inputs: {hh1.inputs}')

def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

def compute_weighted_mpc(c, a, a_grid, r, e_grid):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    mpc[a == a_grid[0]] = 1
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc

def compute_htm(a, a_grid):
    htm = np.zeros_like(a)
    htm[a==a_grid[0]] = 1
    return htm

def compute_consumption(c):
    c01 = np.zeros_like(c)
    c01_10 = np.zeros_like(c)
    c10_35 =np.zeros_like(c)
    c35_65 =np.zeros_like(c)
    c65_90 = np.zeros_like(c)
    c90_99 = np.zeros_like(c)
    c99 = np.zeros_like(c)
    c01[0,:] = c[0,:]
    c01_10[1,:] = c[1,:]
    c10_35[2,:] = c[2,:]
    c35_65[3,:] = c[3,:]
    c65_90[4,:] = c[4,:]
    c90_99[5,:] = c[5,:]
    c99[6,:] = c[6,:]
    return c01, c01_10, c10_35, c35_65, c65_90, c90_99, c99

def compute_assets(a):
    a01 = np.zeros_like(a)
    a01_10 =np.zeros_like(a)
    a10_35 =np.zeros_like(a)
    a35_65 =np.zeros_like(a)
    a65_90 = np.zeros_like(a)
    a90_99 = np.zeros_like(a)
    a99 = np.zeros_like(a)
    a01[0,:] = a[0,:]
    a01_10[1,:] = a[1,:]
    a10_35[2,:] = a[2,:]
    a35_65[3,:] = a[3,:]
    a65_90[4,:] = a[4,:]
    a90_99[5,:] = a[5,:]
    a99[6,:] = a[6,:]
    return a01, a01_10, a10_35, a35_65, a65_90, a90_99, a99

def compute_labor(ne):
    n01 = np.zeros_like(ne)
    n01_10 =np.zeros_like(ne)
    n10_35 =np.zeros_like(ne)
    n35_65 =np.zeros_like(ne)
    n65_90 = np.zeros_like(ne)
    n90_99 = np.zeros_like(ne)
    n99 = np.zeros_like(ne)
    n01[0,:] = ne[0,:]
    n01_10[1,:] = ne[1,:]
    n10_35[2,:] =ne[2,:]
    n35_65[3,:] =ne[3,:]
    n65_90[4,:] = ne[4,:]
    n90_99[5,:] = ne[5,:]
    n99[6,:] = ne[6,:]
    return n01, n01_10, n10_35, n35_65, n65_90, n90_99, n99


hh_ext = hh1.add_hetoutputs([labor_supply, compute_labor,
compute_weighted_mpc, compute_htm, compute_consumption, compute_assets])

print(hh_ext)
print(f'Outputs: {hh_ext.outputs}')


# hh_ext = hh1.add_hetoutputs([labor_supply])

# print(hh_ext)
# print(f'Outputs: {hh_ext.outputs}')


@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

blocks_ss = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc_ss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

print(hank_ss)
print(f"Inputs: {hank_ss.inputs}")


calibration = {'eis': 0.5, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6}

unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0}

ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")


print(f"Asset market clearing: {ss0['asset_mkt']: 0.2e}")
print(f"Labor market clearing: {ss0['labor_mkt']: 0.2e}")
print(f"Goods market clearing (untargeted): {ss0['goods_mkt']: 0.2e}")


# plt.plot(ss0.internals['hh']['a_grid'], ss0.internals['hh']['n'].T)
# plt.xlabel('Assets'), plt.ylabel('Labor supply')
# plt.show()


@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")

print(*hank.blocks, sep='\n')

ss = hank.steady_state(ss0)

for k in ss0.keys():
    assert np.all(np.isclose(ss[k], ss0[k]))


    # setup
T = 300
exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

# general equilibrium jacobians
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

print(G)

#MP shock
rho_r, sig_r = 0.61, -0.01/4
rstar_shock_path = {"rstar": sig_r * rho_r ** (np.arange(T))}

td_nonlin = hank.solve_impulse_nonlinear(ss, unknowns, targets, rstar_shock_path)
td_lin = hank.solve_impulse_linear(ss, unknowns, targets, rstar_shock_path)


dpi_lin=td_lin['pi']
dpi_nonlin=td_nonlin['pi']
# plt.plot(10000 * dpi_lin[:21], label='linear', linestyle='-', linewidth=2.5)
# plt.plot(10000 * dpi_nonlin[:21], label='nonlinear', linestyle='--', linewidth=2.5)
# plt.title(r'Inflation responses monetary policy shocks')
# plt.xlabel('quarters')
# plt.ylabel('bp deviation from ss')
# plt.show()

dc_lin=td_lin['C']
dc_nonlin=td_nonlin['C']
# plt.plot(100 * dc_lin[:21], label='linear', linestyle='-', linewidth=2.5)
# plt.plot(100 * dc_nonlin[:21], label='nonlinear', linestyle='--', linewidth=2.5)
# plt.title(r'Consumption responses monetary policy shocks')
# plt.xlabel('quarters')
# plt.ylabel('absolute deviations from ss')
# plt.show()

fig=plt.figure()
dc_01_10_lin=td_lin['C01_10']
dc_90_99_lin=td_lin['C90_99']
plt.plot(100 * dc_01_10_lin[:50]/ss0['C01_10'], label='linear poorest 10%', linestyle='-', linewidth=2.5)
plt.plot(100 * dc_90_99_lin[:50]/ss0['C90_99'], label='linear richest 10%', linestyle='-', linewidth=2.5)
plt.plot(100 * dc_lin[:50]/ss0['C'], label='linear average', linestyle='-', linewidth=2.5)
plt.title(r'Consumption responses monetary policy shocks')
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()
fig.savefig('Consump_dist_irfs.png')

fig2=plt.figure()
dn_lin=td_lin['N']
dn_01_10_lin=td_lin['N01_10']
dn_90_99_lin=td_lin['N90_99']
plt.plot(100 * dn_01_10_lin[:50]/ss0['N01_10'], label='linear poorest 10%', linestyle='-', linewidth=2.5)
plt.plot(100 * dn_90_99_lin[:50]/ss0['N90_99'], label='linear richest 10%', linestyle='-', linewidth=2.5)
plt.plot(100 * dn_lin[:50]/ss0['N'], label='linear average', linestyle='-', linewidth=2.5)
plt.title(r'Labor responses monetary policy shocks')
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()
fig2.savefig('Labor_dist_irfs.png')