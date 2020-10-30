from pyproj import Transformer
from netCDF4 import Dataset
import numpy as np
import h5py

def get_EASE_grid():

    data_dir = '/home/robbie/Dropbox/Data/IMV/'

    data = Dataset(f'{data_dir}icemotion_daily_nh_25km_20100101_20101231_v4.1.nc')

    lons = np.array(data['longitude'])
    lats = np.array(data['latitude'])

    return(lons, lats)

def kill_empty_rows(tracks_array):

    shape_before = tracks_array.shape

    tracks_array = tracks_array[~np.isnan(tracks_array).any(axis=(1, 2, 3))]

    num_rows_deleted = shape_before[0] - tracks_array.shape[0]

    return(tracks_array, num_rows_deleted)

def extend_tracks_array(tracks_array, velocities):

    time_booster = np.full((velocities.shape[0], tracks_array.shape[1], velocities.shape[3]), np.nan)

    tracks_array = np.concatenate((tracks_array, time_booster), axis=0)

    return(tracks_array)

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

    div_driven_nan_inf_array = np.array([div_driven_nan_inf, div_driven_nan_inf, div_driven_nan_inf])

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

def calculate_div_from_velocities(velocities):

    dudx = np.gradient(velocities[:,:,:,0], axis=1)

    dvdy = np.gradient(velocities[:,:,:,1], axis=2)

    div = np.add(dudx,dvdy)

    return(div)


def get_vectors_for_year(data_dir,year,hemisphere,make_divergence_series):

    data_for_year = Dataset(f'{data_dir}icemotion_daily_{hemisphere}h_25km_{year}0101_{year}1231_v4.1.nc')

    all_u, all_v = np.array(data_for_year['u']), np.array(data_for_year['v'])

    if make_divergence_series:
        velocities = np.stack((all_u, all_v, all_u), axis=3)
    else:
        velocities = np.stack((all_u, all_v), axis=3)

    velocities = np.ma.masked_where(velocities == -9999.0, velocities)
    velocities = np.ma.filled(velocities, np.nan)
    velocities = velocities/100 #Convert cm/s to m/s

    if make_divergence_series:

        div = calculate_div_from_velocities(velocities)

        velocities[:,:,:,2] = div

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

    return 0

def iterate_points(array,
                   velocities_on_day,
                   EASE_tree,
                   timestep,
                   make_divergence_series):


    distances, indexs = EASE_tree.query(array[:,:2])

    if make_divergence_series:

        velocities_of_interest = np.array([velocities_on_day[:, :, 0].ravel()[indexs],
                                           velocities_on_day[:, :, 1].ravel()[indexs],
                                           velocities_on_day[:, :, 2].ravel()[indexs]]).T


        displacements = velocities_of_interest * timestep

        new_positions = array + displacements

        new_positions[:,2] = velocities_of_interest[:,2]


    else:
        velocities_of_interest = np.array([velocities_on_day[:,:,0].ravel()[indexs],
                                           velocities_on_day[:,:,1].ravel()[indexs]]).T

        displacements = velocities_of_interest * timestep

        new_positions = array + displacements

    return (new_positions)


def lonlat_to_xy(coords_1, coords_2, hemisphere, inverse=False):
    """Converts between longitude/latitude and EASE xy coordinates.

    Args:
        lon (float): WGS84 longitude
        lat (float): WGS84 latitude
        hemisphere (string): 'n' or 's'
        inverse (bool): if true, converts xy to lon/lat

    Returns:
        tuple: pair of xy or lon/lat values
    """

    EASE_Proj = {'n': 'epsg:3408',
                 's': 'epsg:3409'}

    WGS_Proj = 'epsg:4326'

    if inverse == False:  # lonlat to xy

        lon, lat = coords_1, coords_2

        transformer = Transformer.from_crs(WGS_Proj, EASE_Proj[hemisphere])

        x, y = transformer.transform(lat, lon)

        return (x, y)

    else:  # xy to lonlat

        x, y = coords_1, coords_2

        transformer = Transformer.from_crs(EASE_Proj[hemisphere], WGS_Proj)

        lat, lon = transformer.transform(x, y)

        return (lon, lat)


