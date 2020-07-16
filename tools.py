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

    all_u = np.ma.masked_where(all_u == -9999.0, all_u)
    all_u = np.ma.filled(all_u, np.nan)
    all_v = np.ma.masked_where(all_v == -9999.0, all_v)
    all_v = np.ma.filled(all_v, np.nan)

    return(all_u, all_v)

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

def one_iteration(point, field, tree, timestep):

    """Iterates a point based on its position in an ice motion field.

    Must be passed a pre-calculated KDTree of the field (which saves time). If the point's nearest velocity value is
    nan (representing open water), then it returns a nan point (np.nan, np.nan)

    Args:
        point: tuple of EASE grid xy coordinates
        field: velocity field in cm/s
        tree: pre-calculated KDTree of grid points of velocity field
        timestep: time in seconds

    Returns:
        Tuple of floats representing xy coordinates of iterated point. If floe disappears then coords are np.nan.

    """

    distance, index = tree.query(point)

    u_vels, v_vels = np.array(field['u']) / 100, np.array(field['v']) / 100

    u_vels = np.ma.masked_where(u_vels == -99.99, u_vels)
    u_vels = np.ma.masked_values(u_vels, np.nan)

    u_vel, v_vel = u_vels.ravel()[index], v_vels.ravel()[index]

    if np.isnan(u_vel):
        #         print('Failed')
        return ((np.nan, np.nan))

    else:

        u_disp, v_disp = u_vel * timestep, v_vel * timestep

        new_position = (point[0] + u_disp, point[1] + v_disp)

        return (new_position)


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

