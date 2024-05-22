import os
import csv
from datetime import datetime, timedelta
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import navpy
import simplekml
import ephemeris_manager
import sol

INPUT_LOG_FILE = r"C:\Users\chai9\OneDrive\שולחן העבודה\A1\Fixed\gnss_log_2024_04_13_19_51_17.txt"
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
def main():

    # Convert log file to CSV file
    with open(INPUT_LOG_FILE) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()
    WEEKSEC = 604800
    LIGHTSPEED = 2.99792458e8

    # This should account for rollovers since it uses a week number specific to each measurement
    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])

    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Convert to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']
    manager = ephemeris_manager.EphemerisManager(ephemeris_data_directory)

    epoch = 0
    num_sats = 0
    while num_sats < 5:
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)

    # Run the function and check out the results:
    sv_position = sol.calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

    # Initial guesses of receiver clock bias and position
    b0 = 0
    x0 = np.array([0, 0, 0])
    xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()

    # Apply satellite clock bias to correct the measured pseudorange values
    pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
    pr = pr.to_numpy()
    x, b, dp = sol.least_squares(xs, pr, x0, b0)

    # Initialize lists to store data
    ecef_coordinates = []
    gps_times = []
    satellite_prns = []
    satellite_x_coords = []
    satellite_y_coords = []
    satellite_z_coords = []
    pseudoranges = []
    cn0_values = []
    doppler_shifts = []

    # Calculate the transmitter location, presented by ECEF #

    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')

        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            satellite_positions = sol.calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

            xs = satellite_positions[['x_k', 'y_k', 'z_k']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * satellite_positions['delT_sv']
            pr = pr.to_numpy()

            x, b, dp = sol.least_squares(xs, pr, x, b)
            for _ in range(len(sats)):
                ecef_coordinates.append(x)

            # Append relevant data to lists
            gps_times.extend(one_epoch['UnixTime'])
            satellite_prns.extend(one_epoch.index)
            satellite_x_coords.extend(satellite_positions['x_k'])
            satellite_y_coords.extend(satellite_positions['y_k'])
            satellite_z_coords.extend(satellite_positions['z_k'])
            pseudoranges.extend(one_epoch['PrM'])
            cn0_values.extend(one_epoch['Cn0DbHz'])
            doppler_shifts.extend(one_epoch['PseudorangeRateMetersPerSecond'])

    # Perform coordinate transformations using the Navpy library
    ECEF = np.stack(ecef_coordinates, axis=0)
    LLA = np.stack(navpy.ecef2lla(ECEF), axis=1)

    # Extract the first position as a reference for the NED transformation
    ref_lla = LLA[0, :]
    ned_array = navpy.ecef2ned(ECEF, ref_lla[0], ref_lla[1], ref_lla[2])

    # Convert back to Pandas and save to csv
    LLA_DF = pd.DataFrame(LLA, columns=['Latitude', 'Longitude', 'Altitude'])
    ECEF_DF = pd.DataFrame(ECEF, columns=['Pos.X', 'Pos.Y', 'Pos.Z'])
    ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

    # Plot
    plt.style.use('dark_background')
    plt.plot(ned_df['E'], ned_df['N'])
    plt.title('Position Offset from First Epoch')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.gca().set_aspect('equal', adjustable='box')

    # Create DataFrame
    plot_data = pd.DataFrame({
        'GPS Time': gps_times,
        'SatPRN (ID)': satellite_prns,
        'Sat.X': satellite_x_coords,
        'Sat.Y': satellite_y_coords,
        'Sat.Z': satellite_z_coords,
        'Pseudo-Range': pseudoranges,
        'CN0': cn0_values,
        'Doppler': doppler_shifts
    })

    # Save DataFrame to CSV
    selected_satellites = plot_data['SatPRN (ID)'].unique().tolist()
    filtered_data = plot_data[plot_data['SatPRN (ID)'].isin(selected_satellites)]

    combined_data = pd.concat([filtered_data, ECEF_DF, LLA_DF], axis=1)
    combined_data.to_csv('output_file.csv', index=False)

    # Create a KML object
    kml = simplekml.Kml()

    kml_locations = []

    for i in range(len(combined_data)):
        location = (combined_data["Latitude"][i]), (combined_data["Longitude"][i])
        kml_locations.append(location)

    # Add each location to the KML file
    for i, (lat, lon) in enumerate(kml_locations):
        kml.newpoint(name=f'Location {i + 1}', coords=[(lon, lat)])

    # Save the KML file
    kml.save("locations.kml")


if __name__ == '__main__':
    main()