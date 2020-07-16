from pyproj import Proj, transform
from netCDF4 import Dataset
import numpy as np
import h5py

def get_EASE_grid():

    data_dir = '/home/robbie/Dropbox/Data/IMV/'

    data = Dataset(f'{data_dir}icemotion_daily_nh_25km_20100101_20101231_v4.1.nc')

    lons = np.array(data['longitude'])
    lats = np.array(data['latitude'])

    return(lons, lats)


def get_vectors_for_year(data_dir,year,hemisphere):

    data_for_year = Dataset(f'{data_dir}icemotion_daily_{hemisphere}h_25km_{year}0101_{year}1231_v4.1.nc')

    all_u, all_v = np.array(data_for_year['u']), np.array(data_for_year['v'])

    velocities = np.stack((all_u, all_v), axis=3)

    velocities = np.ma.masked_where(velocities == -9999.0, velocities)
    velocities = np.ma.filled(velocities, np.nan)

    velocities = velocities/100 #Convert cm/s to m/s


    return(velocities)

def select_and_save_track(track, key, f_name):

    """ Writes floe trajectory to hdf5 file in append mode

    Args:
        track: track coords
        track_no: int representing track number (for later data retrieval)
        f_name: file name of hdf5 storage file

    Returns:
        no return, writes to file.

    """

    with h5py.File(f_name, 'a') as hf:
        hf[f't{key}'] = track

def iterate_points(array,
                   velocities_on_day,
                   EASE_tree,
                   timestep):

    distances, indexs = EASE_tree.query(array)

    velocities_of_interest = np.array([velocities_on_day[:,:,0].ravel()[indexs], velocities_on_day[:,:,1].ravel()[indexs]]).T

    displacements = velocities_of_interest * timestep
    #
    # print('mean displacement:')
    # print(np.nanmean(velocities_of_interest))

    new_positions = array + displacements

    # exit()

    return (new_positions)


def lonlat_to_xy(lon, lat, hemisphere, inverse=False):

    """Converts between longitude/latitude and EASE xy coordinates.

    Args:
        lon (float): WGS84 longitude
        lat (float): WGS84 latitude
        hemisphere (string): 'n' or 's'
        inverse (bool): if true, converts xy to lon/lat

    Returns:
        tuple: pair of xy or lon/lat values
    """

    EASE_Proj_n = Proj(init='epsg:3408')
    EASE_Proj_s = Proj(init='epsg:3409')
    WGS_Proj = Proj(init='epsg:4326')

    EASE_Proj = {'n': EASE_Proj_n,
                 's': EASE_Proj_s}

    if inverse == False:
        x, y = transform(WGS_Proj, EASE_Proj[hemisphere], lon, lat)
        return (x, y)

    else:
        x, y = transform(EASE_Proj[hemisphere], WGS_Proj, lon, lat)
        return (x, y)

