from netCDF4 import Dataset
import datetime
from scipy.spatial import KDTree
import numpy as np
from tools import select_and_save_track, get_vectors_for_year, get_EASE_grid, iterate_points
from tqdm import trange
from tools import lonlat_to_xy
import pickle


def make_daily_tracks():
    """ Makes tracks from daily ice motion vectors

    Script is run from the command line: e.g. python3 main.py 2016 n (for northern hemisphere winter 2016/17). If the
    script isn't run from the command line (for testing, playing), it's automatically configured for 2016 n.

    Returns:
        Nothing (saves file).
    """

    dist_threshold = 25000
    res_factor = 1
    hemisphere = 'n'
    start_year = 2000
    no_years = 17

    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)


    # Get EASE_lons & EASE_lats

    EASE_lons, EASE_lats = get_EASE_grid()

    EASE_xs, EASE_ys = lonlat_to_xy(EASE_lons.ravel(), EASE_lats.ravel(), hemisphere=hemisphere)

    EASE_tree = KDTree(list(zip(EASE_xs, EASE_ys)))

    start_x, start_y = EASE_xs.ravel()[::res_factor], EASE_ys.ravel()[::res_factor]


    # Get dataset for first year

    velocities = get_vectors_for_year(data_dir = '/home/robbie/Dropbox/Data/IMV/',
                                        year = start_year,
                                        hemisphere='n')

    #######################################################

    # Initialise day 1

    data_for_start_day = {'u': velocities[0,:,:,0],
                          'v': velocities[0,:,:,1]}

    u_field = data_for_start_day['u'].ravel()[::res_factor]

    # Select points on ease grid with valid velocity data on day 0

    valid_start_x = start_x[~np.isnan(u_field)]
    valid_start_y = start_y[~np.isnan(u_field)]

    tracks_array = np.full((velocities.shape[0]+50, valid_start_x.shape[0], 2), np.nan)

    tracks_array[0, :, 0] = valid_start_x
    tracks_array[0, :, 1] = valid_start_y

    # for day_num in trange(0, all_u.shape[0]):

    save_key = 0
    start_days = {}
    day_num = 0

    for year in range(start_year, start_year + no_years):

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

            print(f'Day_num: {day_num}, Extant tracks: {tracks_array.shape[1]}')

            # Get the ice motion field for that day

            u_data_for_day = velocities[doy,:,:,0]

            # Update points

            timestep = 24 * 60 * 60

            # Here I want to feed tracks_array[:,day_num,:]
            # I want to recieve a 2d array that gets put into tracks_array[: day_num +1, :]

            # First step is to put all this inside one_iteration function

            updated_points = iterate_points(tracks_array[day_num,:, :],
                                               velocities[doy],
                                               EASE_tree,
                                               timestep)

            # Save these updated points to the numpy array

            tracks_array[day_num+1,:,:] = updated_points

            # Identify index of dead tracks

            dead_cols = np.argwhere(np.isnan(tracks_array[day_num+1,:,0]))

            # Save dead tracks

            for column_no in dead_cols:

                column_no = column_no[0]

                # Find number of non-zero entries in array of x coords

                track_length = np.count_nonzero(~np.isnan(tracks_array[:day_num+1,column_no,0]))

                if track_length > 5:

                    # Start day can be calculated from subtracting the number of extant days from day of death
                    start_day = day_num + 1 - track_length

                    select_and_save_track(tracks_array[start_day:day_num+1,column_no,0],
                                          save_key,
                                          f'long_tracks_{hemisphere}h.h5')

                    start_days[save_key] = (start_day,day_num)

                    save_key+=1


            # Remove dead tracks

            tracks_array = np.delete(tracks_array, dead_cols, axis=1)

            # Make list of points that are still alive

            print(f'Tracks killed: {len(dead_cols)}')

            # Create new parcels in gaps
            # Make a decision tree for the track field

            # Identify all points of ease_grid with valid values

            u_field = np.ma.masked_values(u_data_for_day.ravel()[::res_factor], np.nan)

            valid_grid_points = np.array([start_x[~np.isnan(u_field)],
                                          start_y[~np.isnan(u_field)]]).T


            # Iterate through all valid points to identify gaps using the tree

            if tracks_array.shape[2]:

                track_tree = KDTree(tracks_array[day_num+1,:,:])


                distance, index = track_tree.query(valid_grid_points)

                # Select rows of valid_grid_points where corresponding value in distance array is > dist_threshold

                new_track_initialisations = valid_grid_points[distance>dist_threshold]

                print(f'Tracks added: {new_track_initialisations.shape[0]}')

                # Add newly intitiated tracks to other tracks

                additional_array = np.full((tracks_array.shape[0], new_track_initialisations.shape[0], 2), np.nan)

                additional_array[day_num+1,:,:] = new_track_initialisations

                tracks_array = np.concatenate((tracks_array, additional_array), axis = 1)

            day_num +=1



    # np.save(f'{output_dir}tracks_array_{hemisphere}h_{year}.npy', tracks_array)
    #
    # for track_no in tqdm.trange(tracks_array.shape[2]):
    #     track = tracks_array[:, :, track_no]
    #
    #     select_and_save_track(track,
    #                           track_no,
    #                           f'{output_dir}tracks_{hemisphere}h_{year}.h5')
    #
    # now = datetime.datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("End Time =", current_time)
    #

if __name__ == '__main__':
    make_daily_tracks()