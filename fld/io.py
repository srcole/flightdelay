import os
import pandas as pd
import numpy as np

##
##
##

def load_data(data_folder = '/gh/data/flightdelay/', N_flights = None):
    """Load all airline, airport, and flight data

    Returns
    -------
    df_al : pandas DataFrame
        Airlines data.
    df_ap : pandas DataFrame
        Airports data.
    df_fl : pandas DataFrame
        Flights data.
    """

    df_al = pd.read_csv(os.path.join(data_folder, 'airlines.csv'))
    df_ap = pd.read_csv(os.path.join(data_folder, 'airports.csv'))

    if N_flights is None:
        df_fl = pd.io.parsers.read_csv(data_folder+'flights.csv')
    else:
        df_fl = pd.io.parsers.read_csv(data_folder+'flights.csv', nrows = N_flights)

    return df_al, df_ap, df_fl


def load_data_lines_and_ports(data_folder = '/gh/data/flightdelay/'):
    """Load all airline, airport, and flight data"""
    df_al = pd.DataFrame.from_csv(data_folder+'airlines.csv')
    df_ap = pd.DataFrame.from_csv(data_folder+'airports.csv')

    return df_al, df_ap


def load_data_SAN(data_folder = '/gh/data/flightdelay/',
                    old_data = False,
                    drop_cancelled = True):
    """Load flights departing from SAN"""

    # Load all data
    _, _, df_fl = load_data(data_folder = data_folder)

    # Restrict to SAN data
    restrict = {}
    # If on old data, need to use the 5-digit airport code
    if old_data:
        restrict['ORIGIN_AIRPORT'] = ['SAN','14679',14679]
    else:
        restrict['ORIGIN_AIRPORT'] = ['SAN']

    # Apply restriction
    df_SAN = restrict_df(df_fl, restrict)

    # If needed, remove the cancelled flights
    if drop_cancelled:
        df_SAN = df_SAN[np.isfinite(df_SAN['DEPARTURE_DELAY'])]

    return df_SAN


def restrict_df(df, restriction):
    """Restrict `df` to only the rows in which the key of `restriction`
    takes on one of the values in the associated list"""

    restrict_keys = restriction.keys()

    for k in restrict_keys:
        N_vals = len(restriction[k])
        df_keep = [0]*N_vals
        for i in range(N_vals):
            df_keep[i] = df[df[k]==restriction[k][i]]
        df = pd.concat(df_keep)

    return df


def select_busy_airports(df, n_flights=7300):
    """Restrict DataFrame to only airports with a certain number of outgoing flights."""

    all_airports, airport_inverse, airport_count = np.unique(df['ORIGIN_AIRPORT'],
                                                             return_counts=True,
                                                             return_inverse=True)

    # Determine number of flights for the origin airport
    n_flights_orig = np.zeros(len(airport_inverse))
    for i in range(len(all_airports)):
        n_flights_orig[np.where(airport_inverse==i)] = airport_count[i]

    return df.loc[df.index[n_flights_orig >= n_flights]]


def drop_cancelled(df):
    return df[df['CANCELLED'] != 1]
