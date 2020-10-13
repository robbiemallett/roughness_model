from netCDF4 import Dataset
import datetime
from scipy.spatial import KDTree
import numpy as np
from tools import get_vectors_for_year, get_EASE_grid, iterate_points,\
    remove_dead_tracks, mark_divergence_triggering
from tqdm import trange
from tools import lonlat_to_xy
import pickle


def make_daily_tracks():

    dist_threshold = 25000
    res_factor = 1
    hemisphere = 'n'
    start_year = 2000
    no_years = 1
    printer = False
    save_file_name = f'long_tracks_Oct_testing_bcccz.h5'
    divergence_trigger_check=True


    ################################################################################

    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)


    # Get EASE_lons & EASE_lats

    EASE_lons, EASE_lats = get_EASE_grid()

    EASE_xs, EASE_ys = lonlat_to_xy(EASE_lons.ravel(), EASE_lats.ravel(), hemisphere=hemisphere)

    EASE_tree = KDTree(list(zip(EASE_xs, EASE_ys)))

    start_x, start_y = EASE_xs.ravel()[::res_factor], EASE_ys.ravel()[::res_factor]

    x_index = np.indices(EASE_lons.ravel()[::res_factor].shape)

    # Get dataset for first year

    # This is a 365x361x361x2 array
    # 0 axis is time
    # 1 & 2 axes are ease grid
    # 3 axis just has length two, 0 for u vectors, 1 for v vecotrs

    velocities = get_vectors_for_year(data_dir = '/home/robbie/Dropbox/Data/IMV/',
                                        year = start_year,
                                        hemisphere='n')

    #######################################################

    # Initialise day 1

    # Select first row (day 1), and x and y slices

    data_for_start_day = {'u': velocities[0,:,:,0],
                          'v': velocities[0,:,:,1]}

    u_field = data_for_start_day['u'].ravel()[::res_factor]

    # Select points on ease grid with valid u velocity data on day 0

    valid_start_x = start_x[~np.isnan(u_field)]
    valid_start_y = start_y[~np.isnan(u_field)]

    tracks_array = np.full((velocities.shape[0]+50,
                            valid_start_x.shape[0],
                            2), # Space for x and y coords
                           np.nan)

    tracks_array[0, :, 0] = valid_start_x
    tracks_array[0, :, 1] = valid_start_y

    # for day_num in trange(0, all_u.shape[0]):

    save_key = 0
    start_days = {}
    day_num = 0

    for year in range(start_year, start_year + no_years):
        print(year)

        if year != start_year: #We've already initialised using start_year data, no need to do it again

            pickle.dump(start_days, open('start_days_dict.p','wb'))

            velocities = get_vectors_for_year(data_dir='/home/robbie/Dropbox/Data/IMV/',
                                                year=year,
                                                hemisphere='n')

            # Make the tracks_array longer with each year to accomodate longer and longer tracks

            time_booster = np.full((velocities.shape[0], tracks_array.shape[1], 2), np.nan)

            tracks_array = np.concatenate((tracks_array, time_booster), axis = 0)


        days_in_year = velocities.shape[0]

        for doy in trange(0, days_in_year):

            if printer: print(f'Day_num: {day_num}, Extant tracks: {tracks_array.shape[1]}')

            # Get the ice motion field for that day

            u_data_for_day = velocities[doy,:,:,0]

            # Update points

            timestep = 24 * 60 * 60

            updated_points = iterate_points(tracks_array[day_num,:, :],
                                               velocities[doy],
                                               EASE_tree,
                                               timestep)

            # Save these updated points to the numpy array

            tracks_array[day_num + 1, :, :] = updated_points

            # Identify index of dead tracks

            # Take the x coordinates of tracks_array for the day and look at ones where there's a nan

            tracks_array, save_key, start_days = remove_dead_tracks(tracks_array,
                                                                    save_key,
                                                                    day_num,
                                                                    start_days,
                                                                    save_file_name,
                                                                    printer)

            # Create new parcels in gaps
            # Make a decision tree for the track field

            # Identify all points of ease_grid with valid values

            u_field = u_data_for_day.ravel()[::res_factor]

            valid_grid_points = np.array([start_x[~np.isnan(u_field)],
                                          start_y[~np.isnan(u_field)]]).T


            # Iterate through all valid points to identify gaps using the tree

            # if tracks_array.shape[2]: # Seems like this is always true - test code without?

            track_tree = KDTree(tracks_array[day_num+1,:,:])


            distance, index = track_tree.query(valid_grid_points)

            # Select rows of valid_grid_points where corresponding value in distance array is > dist_threshold

            # A parcel is 'divergence driven' if it's created at an EASE cell where there was also a vector the
            # previous day. Otherwise it's due to creation at the ice edge.

            # Remember 'valid_grid_points' is a 2D array of all x, y coordinates that have valid values on daynum

            new_track_initialisations = valid_grid_points[distance>dist_threshold]

            additional_array = np.full((tracks_array.shape[0], new_track_initialisations.shape[0], 2), np.nan)

            additional_array[day_num+1,:,:] = new_track_initialisations

            if printer: print(f'Tracks added: {new_track_initialisations.shape[0]}')

            if divergence_trigger_check:

                additional_array = mark_divergence_triggering(additional_array,
                                                              x_index,
                                                              u_field,
                                                              distance,
                                                              dist_threshold,
                                                              velocities,
                                                              doy,
                                                              day_num,
                                                              EASE_lons)


            # Add newly intitiated tracks to other tracks

            tracks_array = np.concatenate((tracks_array, additional_array), axis = 1)

            day_num +=1


    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    make_daily_tracks()