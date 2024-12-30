# %%
import duckdb
import pandas as pd

# NOTE: Timezonefinder excluded for now, issues presented during import
# import timezonefinder as tzf
import airportsdata

# Data for this comes from: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL&QO_fu146_anzr=N8vn6v10%20f722146%20gnoyr5
airport_coords = (
    pd.read_csv("../data/T_MASTER_CORD.csv")
    .set_index("AIRPORT")
    .query("AIRPORT_IS_CLOSED < 1 and AIRPORT_IS_LATEST")[
        ["LATITUDE", "LONGITUDE", "DISPLAY_AIRPORT_NAME", "UTC_LOCAL_TIME_VARIATION"]
    ]
)

airport_coords["UTC_LOCAL_TIME_VARIATION"] = airport_coords.UTC_LOCAL_TIME_VARIATION.apply(lambda offset: offset / 100)

# NOTE: Update directory paths as needed
def prep_bts(start_date="2022-12-21", end_date="2022-12-30", table_name="flight_schedule", csv_path="../data/ot_reporting/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2023_1.csv"):
    with duckdb.connect("../data/duck.db") as db:
        db.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")

        flights = db.query(
            f"""
            SELECT * FROM {table_name} schd
            WHERE schd.FlightDate between date '{start_date}' AND date '{end_date}'
            AND schd.Tail_Number IS NOT NULL
            """
        ).to_df()

    # NOTE: UTC offset mapping maintained for testing purposes
    flights["OriginTimezone"] = flights["Origin"].map(
        lambda origin: airport_coords.UTC_LOCAL_TIME_VARIATION[origin])

    flights["DestTimezone"] = flights["Dest"].map(
        lambda origin: airport_coords.UTC_LOCAL_TIME_VARIATION[origin])
    
    flights["Div1Timezone"] = flights["Div1Airport"].map(
        lambda div: None if div is None else airport_coords.UTC_LOCAL_TIME_VARIATION[div])
    
    airports = airportsdata.load('IATA')
    flights["OriginTimezoneName"] = flights["Origin"].map(
        lambda origin: airports[origin]['tz'])
    
    flights["DestTimezoneName"] = flights["Dest"].map(
        lambda dest: airports[dest]['tz'])
    
    flights["Div1TimezoneName"] = flights["Div1Airport"].map(
        lambda div: None if div is None else airports[div]['tz'])

    def dep_time_to_utc(row):
        hour = row["CRSDepTime"] // 100
        minute = row["CRSDepTime"] % 100
        date = row["FlightDate"]
        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute,
            tz=row["OriginTimezoneName"],
        )
        return local_ts.tz_convert("UTC")

    def arr_time_to_utc(row):
        hour = row["CRSArrTime"] // 100
        minute = row["CRSArrTime"] % 100
        
        # account for overnight flights
        date = row["FlightDate"] + (pd.Timedelta(seconds=0) if row["CRSArrTime"] > row["CRSDepTime"] - 400 else pd.Timedelta(days=1))

        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute,
            tz=row["DestTimezoneName"],
        )

        # Handle date line crossing
        # TODO: Verify with respect to extreme long haul flights that do not cross the dateline
        #      (e.g., LAX to DXB or EWR to SIN - the latter of which can be rather complex) 
        if abs(row["TZDifference"]) > 12:
            if row["TZDifference"] > 0:
                local_ts = local_ts + pd.Timedelta(days=1)
            else:
                local_ts = local_ts - pd.Timedelta(days=1)

        return local_ts.tz_convert("UTC")
    
    def div1_arr_time_to_utc(row):
        hour = row["Div1WheelsOn"] // 100
        minute = row["Div1WheelsOn"] % 100
        date = row["FlightDate"]
        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute,
            tz=row["DestTimezone"],
        )
        return local_ts.tz_convert("UTC")

    # def dep_time_to_utc(row):
    #     # Extract hour and minute from CRSDepTime
    #     hour = row["CRSDepTime"] // 100
    #     minute = row["CRSDepTime"] % 100

    #     # Get the flight date
    #     date = row["FlightDate"]

    #     # Compute UTC time directly using the UTC offset
    #     utc_offset = row["OriginTimezone"]  # UTC offset in hours (e.g., -8.0 for PST)
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute
    #     )
    #     # Adjust the local time to UTC
    #     utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
    #     return utc_ts
    
    # def arr_time_to_utc(row):
    #     hour = row["CRSArrTime"] // 100
    #     minute = row["CRSArrTime"] % 100

    #     # Account for overnight flights
    #     date = row["FlightDate"] + (
    #         pd.Timedelta(seconds=0) if row["CRSArrTime"] > row["CRSDepTime"] - 400 else pd.Timedelta(days=1)
    #     )

    #     # Compute UTC time directly using the UTC offset
    #     utc_offset = row["DestTimezone"]  # UTC offset in hours (e.g., -5.0 for EST)
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute
    #     )

    #     # Handle date line crossing
    #     if abs(row["TZDifference"]) > 12:
    #         if row["TZDifference"] > 0:
    #             local_ts = local_ts + pd.Timedelta(days=1)
    #         else:
    #             local_ts = local_ts - pd.Timedelta(days=1)

    #     # Adjust the local time to UTC
    #     utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
    #     return utc_ts

    # def div1_arr_time_to_utc(row):
    #     hour = row["Div1WheelsOn"] // 100
    #     minute = row["Div1WheelsOn"] % 100

    #     date = row["FlightDate"]

    #     # Compute UTC time directly using the UTC offset
    #     utc_offset = row["DestTimezone"]  # UTC offset in hours (e.g., -5.0 for EST)
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute
    #     )
    #     # Adjust the local time to UTC
    #     utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
    #     return utc_ts

    flights["CRSArrTime"] = flights["CRSArrTime"].fillna(0).astype(int)
    flights["CRSDepTime"] = flights["CRSDepTime"].fillna(0).astype(int)
    flights["Div1WheelsOn"] = flights["Div1WheelsOn"].fillna(0).astype(int)

    flights["TZDifference"] = flights["DestTimezone"] - flights["OriginTimezone"]

    flights["ScheduledDepTimeUTC"] = flights.apply(dep_time_to_utc, axis=1)
    flights["ScheduledDepDateUTC"] = flights["ScheduledDepTimeUTC"].dt.date
    flights["ScheduledDepHourUTC"] = flights["ScheduledDepTimeUTC"].dt.hour

    flights["ScheduledArrTimeUTC"] = flights.apply(arr_time_to_utc, axis=1)
    flights["ScheduledArrDateUTC"] = flights["ScheduledArrTimeUTC"].dt.date
    flights["ScheduledArrHourUTC"] = flights["ScheduledArrTimeUTC"].dt.hour

    def get_actual_dep_time(row):
        if pd.isna(row['DepDelay']):
            return pd.NA
        return row['ScheduledDepTimeUTC'] + pd.Timedelta(minutes=row['DepDelay'])
    
    def get_actual_arr_time(row):
        if pd.isna(row['ArrDelay']):
            return pd.NA
        return row['ScheduledArrTimeUTC'] + pd.Timedelta(minutes=row['ArrDelay'])
    
    flights['ActualDepTimeUTC'] = flights.apply(get_actual_dep_time, axis=1)
    flights['ActualArrTimeUTC'] = flights.apply(get_actual_arr_time, axis=1)

    return flights
