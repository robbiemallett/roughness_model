from netCDF4 import Dataset
import datetime
from scipy.spatial import KDTree
import numpy as np
from tools import one_iteration, select_and_save_track, get_vectors_for_year, get_EASE_grid
from tqdm import trange
from tools import lonlat_to_xy


def make_daily_tracks():
    """ Makes tracks from daily ice motion vectors

    Script is run from the command line: e.g. python3 main.py 2016 n (for northern hemisphere winter 2016/17). If the
    script isn't run from the command line (for testing, playing), it's automatically configured for 2016 n.

    Returns:
        Nothing (saves file).
    """

    dist_threshold = 250_000
    res_factor = 10

    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    hemisphere = 'n'
    start_year = 2010

    # Get EASE_lons & EASE_lats

    EASE_lons, EASE_lats = get_EASE_grid()

    EASE_xs, EASE_ys = lonlat_to_xy(EASE_lons.ravel(), EASE_lats.ravel(), hemisphere=hemisphere)

    EASE_tree = KDTree(list(zip(EASE_xs, EASE_ys)))

    start_x, start_y = EASE_xs.ravel()[::res_factor], EASE_ys.ravel()[::res_factor]


    # Get dataset for first year

    all_u, all_v = get_vectors_for_year(data_dir = '/home/robbie/Dropbox/Data/IMV/',
                                        year = start_year,
                                        hemisphere='n')

    #######################################################

    # Initialise day 1

    data_for_start_day = {'u': all_u[0],
                          'v': all_v[0]}

    u_field = data_for_start_day['u'].ravel()[::res_factor]

    # Select points on ease grid with valid velocity data on day 0

    valid_start_x = start_x[~np.isnan(u_field)]
    valid_start_y = start_y[~np.isnan(u_field)]

    valid_points = list(zip(valid_start_x, valid_start_y))

    tracks_array = np.full((2, all_u.shape[0]+50, valid_start_x.shape[0]), np.nan)

    tracks_array[0, 0, :] = valid_start_x
    tracks_array[1, 0, :] = valid_start_y

    # for day_num in trange(0, all_u.shape[0]):

    save_key = 0
    start_days = {}
    day_num = 0

    for year in range(start_year, start_year + 5):

        if year != start_year: #We've already initialised using start_year data, no need to do it again

            all_u, all_v = get_vectors_for_year(data_dir='/home/robbie/Dropbox/Data/IMV/',
                                                year=year,
                                                hemisphere='n')

            # Make the tracks_array longer with each year to accomodate longer and longer tracks

            time_booster = np.full((2, all_u.shape[0], tracks_array.shape[2]), np.nan)

            tracks_array = np.concatenate((tracks_array, time_booster), axis = 1)


        days_in_year = all_u.shape[0]

        for doy in trange(0, days_in_year):


            if day_num > 0:
                valid_points_x = tracks_array[0,day_num,:]
                valid_points_y = tracks_array[1,day_num,:]
                valid_points = list(zip(valid_points_x, valid_points_y))

            print(f'Day_num: {day_num}, Extant tracks: {len(valid_points)}')

            # Get the ice motion field for that day

            data_for_day = {'u': all_u[doy],
                            'v': all_v[doy]}

            # Update points

            updated_points = [one_iteration(point,
                                            data_for_day,
                                            EASE_tree,
                                            24 * 60 * 60) if point != (np.nan, np.nan) else point for point in valid_points]

            # Save these updated points to the numpy array

            tracks_array[0,day_num+1,:] = [coord[0] for coord in updated_points]
            tracks_array[1,day_num+1,:] = [coord[1] for coord in updated_points]

            # Identify index of dead tracks

            dead_cols = np.argwhere(np.isnan(tracks_array[0,day_num+1,:]))

            # Save dead tracks

            for column_no in dead_cols:

                track = tracks_array[:, :day_num+1,column_no]

                # Find number of non-zero entries in array of x coords
                track_length = np.count_nonzero(~np.isnan(track[0]))

                if track_length > 2:

                    # Start day can be calculated from subtracting the number of extant days from day of death
                    start_day = day_num + 1 - track_length

                    select_and_save_track(tracks_array[:,start_day:day_num+1,column_no],
                                          save_key,
                                          f'long_tracks_{hemisphere}h.h5')

                    start_days[save_key] = (start_day,day_num)
                    save_key+=1


            # Remove dead tracks

            tracks_array = np.delete(tracks_array, dead_cols, axis=2)

            # Make list of points that are still alive

            clean_points = [point for point in updated_points if point != (np.nan, np.nan)]

            print(f'Tracks killed: {len(updated_points) - len(clean_points)}')

            # Create new parcels in gaps
            # Make a decision tree for the track field

            # Identify all points of ease_grid with valid values

            u_field = np.ma.masked_values(data_for_day['u'].ravel()[::res_factor], np.nan)

            valid_x, valid_y = start_x[~np.isnan(u_field)], start_y[~np.isnan(u_field)]

            # Iterate through all valid points to identify gaps using the tree

            if clean_points:

                track_tree = KDTree(clean_points)

            new_points = []

            for point in zip(valid_x, valid_y):

                if clean_points == False:
                    new_points.append(point)
                else:

                    distance, index = track_tree.query(point)

                    if (distance > dist_threshold):  # Initiate new track
                        new_points.append(point)

            print(f'Tracks added: {len(new_points)}')

            # Add newly intitiated tracks to other tracks

            additional_array = np.full((2, tracks_array.shape[1], len(new_points)), np.nan)

            additional_array[0,day_num+1,:] = [coord[0] for coord in new_points]
            additional_array[1, day_num+1, :] = [coord[1] for coord in new_points]

            tracks_array = np.concatenate((tracks_array, additional_array), axis = 2)

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