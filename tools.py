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

def mark_divergence_triggering(additional_array,
                              x_index,
                              u_field,
                              distance,
                              dist_threshold,
                              velocities,
                              doy,
                              day_num,
                              EASE_lons):

    valid_grid_index = x_index[0][~np.isnan(u_field)]

    # Get the index of these points
    new_track_indices = valid_grid_index[distance > dist_threshold]

    unraveled_indices = np.unravel_index(new_track_indices, EASE_lons.shape)
    if doy == 0:
        u_data_for_previous = velocities[doy, :, :, 0]
    else:
        u_data_for_previous = velocities[doy - 1, :, :, 0]

    values_for_previous = u_data_for_previous[unraveled_indices]

    # If the values are not nan, then the track is divergence-triggered

    div_driven_bools = ~np.isnan(values_for_previous)

    div_driven_nan_inf = [np.inf if x else np.nan for x in div_driven_bools]

    div_driven_nan_inf_array = np.array([div_driven_nan_inf, div_driven_nan_inf])

    additional_array[day_num, :, :] = div_driven_nan_inf_array.T

    return(additional_array)


def remove_dead_tracks(tracks_array,
                       save_key,
                       day_num,
                       start_days,
                       save_file_name,
                       printer):

    dead_cols = [index[0] for index in np.argwhere(np.isnan(tracks_array[day_num + 1, :, 0]))]

    # deadcols is a list of column indexes that have died.

    # Save dead tracks

    for column_no in dead_cols:

        # Find number of non-zero entries in array of x coords

        track_length = np.count_nonzero(~np.isnan(tracks_array[:, column_no, 0]))

        if track_length > 5:
            # Start day can be calculated from subtracting the number of extant days from day of death
            start_day = day_num - track_length

            # Until recently the function below was only saving x coords.

            select_and_save_track(tracks_array[start_day:day_num + 1, column_no, :],
                                  save_key,
                                  save_file_name)

            start_days[save_key] = {'start_day': start_day,
                                    'day_num': day_num}

            save_key += 1

    # Remove dead tracks

    tracks_array = np.delete(tracks_array, dead_cols, axis=1)

    if printer: print(f'Tracks killed: {len(dead_cols)}')

    return(tracks_array, save_key, start_days)


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

    new_positions = array + displacements

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

