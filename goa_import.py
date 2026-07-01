import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as mcolorbar
from scipy.ndimage import gaussian_filter
import scipy
import cmocean.cm as cmocean
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import random
from tqdm import tqdm

HOME='/oscar/data/deeps/private/ediloren/ebeaudin/'

def anomalies(data,ref_times=slice('1980','2012')):
    clim = data.sel(time=ref_times).groupby('time.month').mean('time')
    return data.groupby('time.month') - clim
    
def coastline(ax=None, lw=1):
    cst = xr.open_dataset('/oscar/data/deeps/private/ediloren/ebeaudin/data/WorldCoastline_Pacific.nc')
    if ax is not None:
        ax.plot(cst.clon, cst.clat, 'k', linewidth=lw)
        ax.set_xlim(-170, -122); ax.set_ylim(45, 62)
    else:
        plt.plot(cst.clon, cst.clat, 'k', linewidth=lw)
        plt.xlim(-170, -122); plt.ylim(45, 62)
        
def coastline_cartopy(ax):
    ax.add_feature(cfeature.LAND, zorder=10, facecolor='w')
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, color='k', linewidth=1.2, zorder=102)
    ax.plot([0, 0], [0, 1], transform=ax.transAxes, color='k', linewidth=1.2, zorder=102)
    
    # Add lat/lon gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0., color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

def smooth(data, size=1, sigma=1, method='convolve2d', flood=True):
    ### --- flood function ---
    def fillna_nearest_2d(da):
        from scipy.ndimage import distance_transform_edt
        """Fill NaNs in 2D DataArray with nearest-neighbor interpolation."""
        if da.ndim != 2:
            raise ValueError("Only works on 2D DataArray (e.g., lat x lon)")
    
        data = da.values
        mask = np.isnan(data)
    
        # Get nearest neighbor indices
        idx = np.array(np.meshgrid(*[np.arange(s) for s in data.shape], indexing='ij'))
        dist, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    
        filled = data[tuple(nearest_idx)]
        filled = xr.DataArray(filled, coords=da.coords, dims=da.dims, attrs=da.attrs)
        mask = xr.DataArray(mask, coords=da.coords, dims=da.dims, attrs=da.attrs)
        return filled, mask
    ### ----------------------
    
    from scipy.ndimage import gaussian_filter
    
    if not isinstance(data, xr.DataArray):
        raise ValueError("Input must be an xarray.DataArray")

    # Fill nans first
    if flood:
        data, mask = fillna_nearest_2d(data)
    else: mask = np.isnan(data)
        
    if method == 'gaussian':
        smoothed = gaussian_filter(data.values, sigma)
    elif method == 'convolve2d':
        from scipy.signal import convolve2d
        x = np.linspace(-size, size, 2 * size + 1)
        y = np.linspace(-size, size, 2 * size + 1)
        x, y = np.meshgrid(x, y)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        smoothed = convolve2d(data.values, kernel, mode='same', boundary='symm')
    else:
        raise ValueError("Unsupported method. Choose 'gaussian' or 'convolve2d'.")
    
    smoothed_da = xr.DataArray(smoothed, coords=data.coords, dims=data.dims, attrs=data.attrs)
    
    return smoothed_da.where(~mask)


# EOF Function

from scipy.linalg import svd

# remap function
def remap(y,I,J,ocean_points):
    y_map = np.empty([I*J])*np.nan
    y_map[ocean_points] = y
    return np.reshape(y_map,(I,J))
    
def EofsAnalysis(ano,n):
    # Function that computes first n eofs and pcs of the signal
    # ano: anomalies (lon,lat,time)
    # n: number of eofs
    
    # Reshape data [I*J,T]
    [I,J,T] = ano.shape # get dataset dimensions
    y = np.reshape(ano,[I*J,T]) # reshape dataset (I,J,T) >> (I*J,T)
    ocean_points = np.where(~np.isnan(y[:,0]))[0] # find ocean points
    y = y[ocean_points,:] # take ocean points only
    
    # Covariance matrix
    Cyy = np.dot(y,y.T)/T
    
    # Compute eigenvalues (l) and eigenvectors (U)
    [l,U] = np.linalg.eig(Cyy)
    l = np.real(l) #eigenvalues
    L = np.diag(l) #matrix of l
    U = np.real(U) #eigenvectors
    
    # Variance explained
    varexp = l/l.sum()*100
    N=12 #d.f.
    dlmbda = varexp*np.sqrt(2/N) #errorbars
    varexp = varexp[0:n] #keep the first n only
    dlmbda = dlmbda[0:n]
    
    # Eofs
    Unorm = U*np.sqrt(T-1) #normalize
    eofs = np.empty([I,J,n])
    for i in range(n):
        eofs[...,i] = remap(Unorm[:,i],I,J,ocean_points) #remap for plotting
    
    # Pcs
    P = np.dot(U.T,y)
    Pnorm = P/np.sqrt(T-1) #normalize
    pcs = Pnorm[0:n,:]

    # Inverse signs
    eofs*=-1
    pcs*=-1
    
    return eofs, pcs, varexp, dlmbda

def plot_variance_explained(varexp):
    # Variance explained
    k=np.arange(1,len(varexp)+1,1)
    plt.figure(figsize=(6,2))
    plt.plot(k,varexp,'-o', color='darkblue', label='individual',zorder=100) #variance expplained
    plt.errorbar(x=k, y=varexp, yerr=dlmbda, capthick=20, color='darkblue') #errorbars
    plt.plot(k,varexp.cumsum(),'--o', color='darkred',label='cum. sum') #cummulative sum
    plt.grid()
    plt.title('Variance explained',fontsize=16)
    plt.xlabel('mode',fontsize=14)
    plt.ylabel('(%)',fontsize=14)
    plt.legend()
    plt.show();


def plot_first_eofs(eofs,varexp,pcs,lat,lon,time,title='',cmap='RdYlBu_r'):
    fig, ax = plt.subplots(1,3,figsize=(12,2), sharey=True)

    # EOFs
    for n in range(3):
        p=ax[n].contourf(lon,lat,eofs[:,:,n],np.arange(-1,1.2,.2), cmap=cmap, extend='both')
        coastline(ax=ax[n],region='GOA')
        ax[n].set_title(f'EOF {n+1} ({varexp[n]:.1f} %)')
    fig.colorbar(p,ax=ax)
    plt.show()

    # PCs
    colors=['darkblue','orange','purple']
    plt.figure(figsize=(12,2))
    for n in range(3):
        plt.plot(time, pcs[n,:], c=colors[n], linewidth=2-n*0.3, label=f'PC {n+1}')
    
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.grid(color='g',alpha=0.1)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

def detrend_data(ds, dim='time'):
    """Detrends data along a specified dimension while handling NaNs."""
    def detrend_1d(y):
        x = np.arange(len(y))
        mask = ~np.isnan(y)  # Only use non-NaN values for fitting
        if mask.sum() > 1:  # Ensure enough points to fit a trend
            p = np.polyfit(x[mask], y[mask], 1)  # Linear detrend
            y_detrended = y - np.polyval(p, x)
        else:
            y_detrended = y  # Not enough points, return unchanged
        return y_detrended

    return xr.apply_ufunc(
        detrend_1d,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized"
    )

# EOF Dataset
def get_eofs(y_ano, n=5, plot=0, cmap='RdYlBu_r'):
    y_ano = y_ano.fillna(0)
    y_ano_detrended = detrend_data(y_ano, dim='time') # Detrend data
    y_ano = y_ano.transpose('longitude', 'latitude', 'time') # Transpose to [Lon, Lat, Time]
    eofs, pcs, varexp, dlmda = EofsAnalysis(y_ano.values,n) # Compute first 10 eofs/pcs
    
    
    # Data Array
    nmodes=eofs.shape[-1]
    
    eofs_dataset = xr.Dataset(
    {
        "eofs": xr.DataArray(
            eofs, coords={"longitude": y_ano.longitude, "latitude": y_ano.latitude, "mode": np.arange(nmodes)}, dims=["longitude", "latitude", "mode"]
        ),
        "pcs": xr.DataArray(
            pcs, coords={"mode": np.arange(nmodes), "time": y_ano.time.coords["time"]}, dims=["mode", "time"]
        ),
        "varexp": xr.DataArray(
            varexp, coords={"mode": np.arange(nmodes)}, dims=["mode"]
        ),
        "dlmda": xr.DataArray(
            dlmda, coords={"mode": np.arange(nmodes)}, dims=["mode"]
        )        
    }
    )
    
    if plot==1:
        # Plot EOFs and PCs
        #plot_variance_explained(varexp)
        plot_first_eofs(eofs_dataset.eofs,eofs_dataset.varexp,eofs_dataset.pcs,y_ano['lat_rho'],y_ano['lon_rho'],y_ano['time'],title=varname, cmap=cmap)

    return eofs_dataset

def compute_eke(u, v):

    u_mean = u.sel(time=slice('1993','2012')).mean(dim='time')
    v_mean = v.sel(time=slice('1993','2012')).mean(dim='time')

    #u_prime = u - u_mean
    #v_prime = v - v_mean
    
    u_prime = anomalies(u)
    v_prime = anomalies(v)
    
    eke = 0.5 * ((u_prime.values**2) + (v_prime.values**2))
    
    return xr.DataArray(eke, coords=u.coords, dims=u.dims)

def compute_eke_geos(ssh):
    lat, lon = ssh.latitude, ssh.longitude
    lons, lats = np.meshgrid(lon,lat)
    
    # Convert lat/lon to radians
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    
    # Constants
    R = 6371000  # Earth's radiusin meters
    g = 9.81  # gravitational acceleration (m/s^2)
    omega = 7.2921e-5  # Earth's rotation rate, rad/s
    f = 2 * omega * np.sin(lat_rad)
    
    # Calculate the differences in lat and lon (assumed to be uniform across time)
    dlat = np.gradient(lat_rad, axis=0)  # change in lat
    dlon = np.gradient(lon_rad, axis=1)  # change in lon
    
    # Calculate the spacing in meters
    dx = R * np.cos(lat_rad) * dlon  # zonal distance
    dy = R * dlat                    # meridional distance
    
    # Initialize arrays for u_g and v_g
    u_g = np.empty_like(ssh)
    v_g = np.empty_like(ssh)
    
    # Loop over ocean_time to compute the gradients and velocities
    for t in range(ssh.shape[0]):
        dzeta_dy = np.gradient(ssh[t, ...], axis=0) / dy
        dzeta_dx = np.gradient(ssh[t, ...], axis=1) / dx

        u_g[t, ...] = -(g / f) * dzeta_dy
        v_g[t, ...] = (g / f) * dzeta_dx
    
    u_prime = u_g - np.nanmean(u_g,0)
    v_prime = v_g - np.nanmean(v_g,0)
    
    # Calculate GEOS EKE
    eke = 0.5 * (u_prime**2 + v_prime**2)
    
    return xr.DataArray(eke, coords=ssh.coords, dims=ssh.dims)

def compute_u_geos(ssh):
    lat, lon = ssh.latitude, ssh.longitude
    lons, lats = np.meshgrid(lon,lat)
    
    # Convert lat/lon to radians
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    
    # Constants
    R = 6371000  # Earth's radiusin meters
    g = 9.81  # gravitational acceleration (m/s^2)
    omega = 7.2921e-5  # Earth's rotation rate, rad/s
    f = 2 * omega * np.sin(lat_rad)
    
    # Calculate the differences in lat and lon (assumed to be uniform across time)
    dlat = np.gradient(lat_rad, axis=0)  # change in lat
    dlon = np.gradient(lon_rad, axis=1)  # change in lon
    
    # Calculate the spacing in meters
    dx = R * np.cos(lat_rad) * dlon  # zonal distance
    dy = R * dlat                    # meridional distance
    
    # Initialize arrays for u_g and v_g
    u_g = np.empty_like(ssh)
    v_g = np.empty_like(ssh)

    # Loop over ocean_time to compute the gradients and velocities
    for t in range(ssh.shape[0]):
        dzeta_dy = np.gradient(ssh[t, ...], axis=0) / dy
        dzeta_dx = np.gradient(ssh[t, ...], axis=1) / dx

        u_g[t, ...] = -(g / f) * dzeta_dy
        v_g[t, ...] = (g / f) * dzeta_dx
        
    geo_dataset = xr.Dataset(
        {
            "ugeo": xr.DataArray(u_g, coords=ssh.coords, dims=ssh.dims),
            "vgeo": xr.DataArray(v_g, coords=ssh.coords, dims=ssh.dims),   
        }
    )
    return geo_dataset


def fillna_nearest_2d(da):
    from scipy.ndimage import distance_transform_edt
    """Fill NaNs in 2D DataArray with nearest-neighbor interpolation."""
    if da.ndim != 2:
        raise ValueError("Only works on 2D DataArray (e.g., lat x lon)")

    data = da.values
    mask = np.isnan(data)

    # Get nearest neighbor indices
    idx = np.array(np.meshgrid(*[np.arange(s) for s in data.shape], indexing='ij'))
    dist, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)

    filled = data[tuple(nearest_idx)]
    return xr.DataArray(filled, coords=da.coords, dims=da.dims, attrs=da.attrs)

# Transpose signal into frequency space, using Fourier Series

def fourier(signal):
    from scipy.fftpack import fft
    
    sampling_rate = 1 #months
    N = np.size(signal)
    f = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/sampling_rate)

    nf = f/f.max() # normalized spectrum
    power = 2./N * np.abs(f[0:N//2]) #spectrum to plot
    power = power / np.sum(power) # normalized spectrum
    
    return freqs, power, N

def red_noise(ts, alpha=0.9, beta=1.0):
    from scipy.stats import f
    
    # Parameters
    T = len(ts)
    #alpha = 0.9 # auto-correlation [0=white noise, 1=random walk]
    #beta = 1.0 # noise amplitude/std [1=normal std]
    
    # Generate red noise spectrum (e.g. average over 100 synthetic series)
    n_sim = 100
    spec_sim = []
    
    for _ in range(n_sim):
        x = np.zeros(T)
        x[0] = beta*np.random.randn()
        for j in range(1, T):
            x[j] = alpha * x[j-1] + beta * np.random.randn()
        f_syn = np.fft.fft(x)
        p_syn = 2./T * np.abs(f_syn[0:T//2])
        spec_sim.append(p_syn)
    
    # Average red-noise spectrum
    rspec = np.mean(spec_sim, axis=0)
    rspec = rspec/ np.sum(rspec) # normalize
    
    # F-test significance
    dof = 2
    fstat99 = f.ppf(0.99, dof, 1000)
    spec99 = rspec * fstat99
    
    fstat95 = f.ppf(0.95, dof, 1000)
    spec95 = rspec * fstat95

    fstat90 = f.ppf(0.90, dof, 1000)
    spec90 = rspec * fstat90

    return rspec, spec90, spec95, spec99
