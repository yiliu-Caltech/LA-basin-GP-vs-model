#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:39:42 2024

@author: glavrent
"""
#load libraries
import os
import pathlib
import sys
#arithmetic libraries
import numpy  as np
import pandas as pd
#ploting libraries
import matplotlib.pyplot as plt
#user functions
sys.path.insert(0,'../python_lib/statistics')
sys.path.insert(0,'../python_lib/vel_prof')
sys.path.insert(0,'../python_lib/plotting')
sys.path.insert(0,'../implementation')
sys.path.insert(0,'../miscellaneous')
sys.path.insert(0,'../python_lib/usgs')
#plotting
import pylib_contour_plots as pycplt
#vs30 map
from query_willis15_ca_vs30_map import Willis15Vs30CA as willis15vs30
#usgs vel model
from query_usgs_v21 import USGSVelModelv21 as velmodel_usgs
#velocity model
from vel_model import VelModelStationary as velmodel_stat
from vel_model import VelModelSpatialVarying as velmodel_svar

#%% Define Variables
### ======================================
#velocity model exents
# vel_model_ext = np.array( [[-120.644051, -121.922036, -123.858493, -122.562365, -120.644051],
#                            [37.050062,    36.320331,   38.424179,   39.174505,   37.050062]])
#velocity model exents (~1% shrinked)
vel_model_ext = np.array([[-120.7085957 , -121.8574913 , -123.79303199, -122.62782601, -120.7085957 ],
                          [  37.0132069 ,   36.3571861 ,   38.46207425,   39.13660975,   37.0132069 ]])

#number points
# n_discrt = 1000
n_discrt = 500
# n_discrt = 100

#depth
# z = np.array([0.,])
# z = np.array([10.,])
# z = np.array([30.,])
z = np.array([50.,])
# z = np.array([100.,])

#color bar limits
if z[0] == 0:
    lims_vs  = [50, 1000]
    lims_unc = [0., .10]
elif z[0] == 10:
    lims_vs  = [50, 1200]
    lims_unc = [0., .10]    
elif z[0] == 30:
    lims_vs  = [200, 1300]
    lims_unc = [0., .10]
elif z[0] == 50:
    lims_vs  = [250, 1500]
    lims_unc = [0., .13]
elif z[0] == 100:
    lims_vs  = [400, 1600]
    lims_unc = [0., .15]

#filename spatially varying parametes
fname_model_svar_dBr = '../../Data/regression/model_spatially_varying_old/all_trunc_stan_parameters2.csv'

#usgs vel model
dir_usgs   = '/mnt/halcloud_nfs/glavrent/Research/GP_Vel_profiles/Raw_files/vel_model/USGS_SFB_vel_model/'
dir_usgs   = '/home/glavrent/Downloads/'
fname_usgs = 'USGS_SFCVM_v21-1_detailed.h5'

#figure directory
dir_fig = '../../Data/misc/comparison_maps/'

#main figure title
fname_fig_main = 'map_original_'

#%% Load Data
### ======================================
#read velocity profile locations
df_model_svar_param = pd.read_csv(fname_model_svar_dBr)

#velocity profile locations for plotting
vel_prof_loc = df_model_svar_param.loc[:,['DSID','VelID','Lon','Lat']]
#remove profiles outside the domain
vel_prof_loc = vel_prof_loc.loc[~np.logical_and(vel_prof_loc.DSID==1,vel_prof_loc.VelID==6),:]
vel_prof_loc = vel_prof_loc.loc[~np.logical_and(vel_prof_loc.DSID==3,vel_prof_loc.VelID==69),:]

#%% Processing
### ======================================
#edge coordinates
edge1_xy = np.linspace(vel_model_ext[:,0], vel_model_ext[:,1], n_discrt)
edge2_xy = np.linspace(vel_model_ext[:,3], vel_model_ext[:,2], n_discrt)

#create grid
mesh_grid = []
#iterate over edge points
for e1_xy, e2_xy in zip(edge1_xy, edge2_xy):
    #create new layer
    mesh_grid.append(np.linspace(e1_xy, e2_xy, n_discrt))

#combine layers 
mesh_grid = np.array(mesh_grid)
mesh_array = np.reshape(mesh_grid, [np.prod(mesh_grid.shape[:2]), 2]  )

#add observation locations
mesh_array = np.vstack([mesh_array, vel_prof_loc.loc[:,['Lon','Lat']]])

#number of points
n_pt = mesh_array.shape[0]


# Evaluate Vs30
# --------------------------------
model_w15ca_vs30 = willis15vs30()
vs30_mu_array, vs30_sd_array = model_w15ca_vs30.lookup( mesh_array )

# Empirical velocity models
# --------------------------------
#stationary velocity model
model_vel_stat = velmodel_stat()
stat_vs_mu_array, stat_vs_sd_array = model_vel_stat.Vs(vs30_mu_array, z)

#spatially varying velocity model (conidtional)
model_vel_svar_cnd = velmodel_svar(fname_dBr=fname_model_svar_dBr)
# svar_vs_mu_array, svar_vs_glb_array, svar_vs_sd_array = model_vel_svar.Vs(vs30_mu_array, z, latlon=np.fliplr(mesh_array))
# svar_vs_unc_array = model_vel_svar.Vs(vs30_mu_array, z, latlon=np.fliplr(mesh_array), n_realiz=100)[0]
# svar_vs_unc_array = np.std( np.log( model_vel_svar.Vs(vs30_mu_array, z, latlon=np.fliplr(mesh_array), n_realiz=250)[0] ), axis=2)

#initialize arrays
svar_cnd_vs_mu_array  = np.full(stat_vs_mu_array.shape, np.nan)
svar_cnd_vs_glb_array = np.full(stat_vs_mu_array.shape, np.nan)
svar_cnd_vs_sd_array  = np.full(stat_vs_mu_array.shape, np.nan)
svar_cnd_vs_unc_array = np.full(stat_vs_mu_array.shape, np.nan)
#iterate over all grid points
idx_all = np.array_split(np.arange(n_pt), np.ceil(n_pt/200))
for j, idx in enumerate( idx_all ):
    print('evaluating conditional spatially varying vel batch: %i of %i ...'%(j+1,len(idx_all)))
    #evaluate spatially varying model
    svar_cnd_vs_mu_array[0,idx], svar_cnd_vs_glb_array[0,idx], svar_cnd_vs_sd_array[0,idx] = model_vel_svar_cnd.Vs(vs30_mu_array[idx], z, latlon=np.fliplr(mesh_array)[idx])
    svar_cnd_vs_unc_array[0,idx] = np.std( np.log( model_vel_svar_cnd.Vs(vs30_mu_array[idx], z, latlon=np.fliplr(mesh_array)[idx], n_realiz=250)[0] ), axis=2)

#spatially varying velocity model (unconidtional)
model_vel_svar_ucnd = velmodel_svar()

#initialize arrays
svar_ucnd_vs_mu_array  = np.full(stat_vs_mu_array.shape, np.nan)
svar_ucnd_vs_glb_array = np.full(stat_vs_mu_array.shape, np.nan)
svar_ucnd_vs_sd_array  = np.full(stat_vs_mu_array.shape, np.nan)
svar_ucnd_vs_unc_array = np.full(stat_vs_mu_array.shape, np.nan)
#iterate over all grid points
idx_all = np.array_split(np.arange(n_pt), np.ceil(n_pt/200))
for j, idx in enumerate( idx_all ):
    print('evaluating unconditional spatially varying vel batch: %i of %i ...'%(j+1,len(idx_all)))
    #evaluate spatially varying model
    svar_ucnd_vs_mu_array[0,idx], svar_ucnd_vs_glb_array[0,idx], svar_ucnd_vs_sd_array[0,idx] = model_vel_svar_ucnd.Vs(vs30_mu_array[idx], z, latlon=np.fliplr(mesh_array)[idx])
    svar_ucnd_vs_unc_array[0,idx] = np.std( np.log( model_vel_svar_ucnd.Vs(vs30_mu_array[idx], z, latlon=np.fliplr(mesh_array)[idx], n_realiz=250)[0] ), axis=2)

# USGS model
# --------------------------------
model_vel_usgs = velmodel_usgs(dir_usgs, fname_usgs )

#initialize arrays
usgs_vs_array  = np.full(stat_vs_mu_array.shape, np.nan)

#iterate over all locations
for j in range(n_pt):
    if j == 0 or not (j+1)%100 or j+1==n_pt : print('evaluating USGS vel prof: %i of %i ...'%(j+1,n_pt))
    usgs_vs_array[0,j] = model_vel_usgs.QueryZ(np.fliplr(mesh_array)[j], z=z)[2].Vs.values[0]

#%% Plotting
### ======================================
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

# Spatial variability
# ---------------------------
# pl_win = [[36, 39], [-124, -121]]
pl_win = [[36.25, 39.25], [-123.75, -120.5]]

#veloicty model 
vel_model_edges = [np.linspace(vel_model_ext[:,j-1], vel_model_ext[:,j], n_discrt) for j in range(vel_model_ext.shape[1])]
vel_model_edges = np.vstack( vel_model_edges )

# Stationary model
# ---   ---   ---   ---   ---
stat_vs_data2plot = np.hstack([np.fliplr(mesh_array), stat_vs_mu_array.T]) 
stat_vs_data2plot = stat_vs_data2plot[~np.isnan(stat_vs_data2plot[:,2]),:]
stat_vs_data2plot = stat_vs_data2plot[stat_vs_data2plot[:,2]>0,:]

fname_fig =  'stationary_model_med_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(stat_vs_data2plot, 
                                                      cmin=np.log(lims_vs[0]), cmax=np.log(lims_vs[1]), log_cbar=True, frmt_clb='%.1f')
# ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=4)
ax.plot(vel_model_ext[0,:],vel_model_ext[1,:],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$V_S[z=%.1fm]$ (m/sec)'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

# Spatial varying model
# ---   ---   ---   ---   ---
svar_cnd_vs_data2plot = np.hstack([np.fliplr(mesh_array), svar_cnd_vs_mu_array.T, svar_cnd_vs_unc_array.T]) 
svar_cnd_vs_data2plot = svar_cnd_vs_data2plot[~np.isnan(svar_cnd_vs_data2plot[:,2]),:]
svar_cnd_vs_data2plot = svar_cnd_vs_data2plot[svar_cnd_vs_data2plot[:,2]>0,:]

fname_fig =  'spatially_varying_model_conditional_med_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(svar_cnd_vs_data2plot[:,[0,1,2]], 
                                                      cmin=np.log(lims_vs[0]), cmax=np.log(lims_vs[1]), log_cbar=True, frmt_clb='%.1f')
ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=1)
ax.plot(vel_model_edges[:,0],vel_model_edges[:,1],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$V_S[z=%.1fm]$ (m/sec)'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

fname_fig =  'spatially_varying_model_conditional_unc_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(svar_cnd_vs_data2plot[:,[0,1,3]], 
                                                      cmin=lims_unc[0], cmax=lims_unc[1], log_cbar=False, frmt_clb='%.2f')
ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=1)
ax.plot(vel_model_ext[0,:],vel_model_ext[1,:],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$\psi[z=%.1fm]$ (log(m/sec))'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

# Spatial varying model
# ---   ---   ---   ---   ---
svar_ucnd_vs_data2plot = np.hstack([np.fliplr(mesh_array), svar_ucnd_vs_mu_array.T, svar_ucnd_vs_unc_array.T]) 
svar_ucnd_vs_data2plot = svar_ucnd_vs_data2plot[~np.isnan(svar_ucnd_vs_data2plot[:,2]),:]
svar_ucnd_vs_data2plot = svar_ucnd_vs_data2plot[svar_ucnd_vs_data2plot[:,2]>0,:]

fname_fig = 'spatially_varying_model_unconditional_med_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(svar_ucnd_vs_data2plot[:,[0,1,2]], 
                                                      cmin=np.log(lims_vs[0]), cmax=np.log(lims_vs[1]), log_cbar=True, frmt_clb='%.1f')
# ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=1)
ax.plot(vel_model_edges[:,0],vel_model_edges[:,1],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$V_S[z=%.1fm]$ (m/sec)'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

fname_fig = 'spatially_varying_model_unconditional_unc_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(svar_ucnd_vs_data2plot[:,[0,1,3]], 
                                                      cmin=lims_unc[0], cmax=lims_unc[1], log_cbar=False, frmt_clb='%.2f')
# ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=1)
ax.plot(vel_model_ext[0,:],vel_model_ext[1,:],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$\psi[z=%.1fm]$ (log(m/sec))'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

# USGS Model
# ---   ---   ---   ---   ---
usgs_vs_data2plot = np.hstack([np.fliplr(mesh_array), usgs_vs_array.T]) 
usgs_vs_data2plot = usgs_vs_data2plot[~np.isnan(usgs_vs_data2plot[:,2]),:]
usgs_vs_data2plot = usgs_vs_data2plot[usgs_vs_data2plot[:,2]>0,:]

fname_fig =  'usgs_model_z_%.1fm'%z[0]
fig, ax, cbar, data_crs, gl = pycplt.PlotContourCAMap(usgs_vs_data2plot, 
                                                      cmin=np.log(lims_vs[0]), cmax=np.log(lims_vs[1]), log_cbar=True, frmt_clb='%.1f')
# ax.plot(df_model_svar_param.Lon.values, df_model_svar_param.Lat.values,'ok',linewidth=3, zorder=5, markersize=4)
ax.plot(vel_model_ext[0,:],vel_model_ext[1,:],'-k',linewidth=3, zorder=5)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$V_S[z=%.1fm]$ (m/sec)'%z, size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig_main + fname_fig + '.png' )

