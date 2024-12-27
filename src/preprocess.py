# %%
import duckdb
import pandas as pd

# TODO: Timezonefinder is no longer needed - BTS data now includes timezone information
# import timezonefinder as tzf

# Data for this comes from: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL&QO_fu146_anzr=N8vn6v10%20f722146%20gnoyr5
airport_coords = (
    pd.read_csv("../data/T_MASTER_CORD.csv")
    .set_index("AIRPORT")
    .query("AIRPORT_IS_CLOSED < 1 and AIRPORT_IS_LATEST")[
        ["LATITUDE", "LONGITUDE", "DISPLAY_AIRPORT_NAME", "UTC_LOCAL_TIME_VARIATION"]
    ]
)

airport_coords["UTC_LOCAL_TIME_VARIATION"] = airport_coords.UTC_LOCAL_TIME_VARIATION.apply(lambda offset: offset / 100)

# TODO: Update directory paths as needed
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


        # flights = db.query(
        #     f"""
        #     select * from {table_name} mess
        #     left join (
        #         select id, list(odc_level) as levels from southwest_odcs group by id
        #     ) odcs on mess.column000 = odcs.id
        #     where mess.FlightDate between date '{start_date}' and date '{end_date}' and
        #     mess.column000 is not null"""
        # ).to_df()
        # finder = tzf.TimezoneFinder()

    # TODO: Skip the use of timezonefinder and directly use the timezone data in the BTS data; make sure to correct by dividing by 100
    # flights["OriginTimezone"] = flights["Origin"].map(
    #     lambda origin: finder.timezone_at(
    #         lng=airport_coords.LONGITUDE[origin], lat=airport_coords.LATITUDE[origin]
    #     )
    # )
    # flights["DestTimezone"] = flights["Dest"].map(
    #     lambda origin: finder.timezone_at(
    #         lng=airport_coords.LONGITUDE[origin], lat=airport_coords.LATITUDE[origin]
    #     )
    # )
    # flights["Div1Timezone"] = flights["Div1Airport"].map(
    #     lambda div: None if div is None else finder.timezone_at(
    #         lng=airport_coords.LONGITUDE[div], lat=airport_coords.LATITUDE[div]
    #     )
    # )

    flights["OriginTimezone"] = flights["Origin"].map(
        lambda origin: airport_coords.UTC_LOCAL_TIME_VARIATION[origin])

    flights["DestTimezone"] = flights["Dest"].map(
        lambda origin: airport_coords.UTC_LOCAL_TIME_VARIATION[origin])
    
    flights["Div1Timezone"] = flights["Div1Airport"].map(
        lambda div: None if div is None else airport_coords.UTC_LOCAL_TIME_VARIATION[div])

    # TODO: As such these will need to be updated to handle the UTC offset directly from the data
    # def dep_time_to_utc(row):
    #     hour = row["CRSDepTime"] // 100
    #     minute = row["CRSDepTime"] % 100
    #     date = row["FlightDate"]
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute,
    #         tz=row["OriginTimezone"],
    #     )
    #     return local_ts.tz_convert("UTC")

    # def arr_time_to_utc(row):
    #     hour = row["CRSArrTime"] // 100
    #     minute = row["CRSArrTime"] % 100
    #     # account for overnight flights
    #     date = row["FlightDate"] + (pd.Timedelta(seconds=0) if row["CRSArrTime"] > row["CRSDepTime"] - 400 else pd.Timedelta(days=1))
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute,
    #         tz=row["DestTimezone"],
    #     )
    #     return local_ts.tz_convert("UTC")
    
    # def div1_arr_time_to_utc(row):
    #     hour = row["Div1WheelsOn"] // 100
    #     minute = row["Div1WheelsOn"] % 100
    #     date = row["FlightDate"]
    #     local_ts = pd.Timestamp(
    #         year=date.year,
    #         month=date.month,
    #         day=date.day,
    #         hour=hour,
    #         minute=minute,
    #         tz=row["DestTimezone"],
    #     )
    #     return local_ts.tz_convert("UTC")

    def dep_time_to_utc(row):
        # Extract hour and minute from CRSDepTime
        hour = row["CRSDepTime"] // 100
        minute = row["CRSDepTime"] % 100

        # Get the flight date
        date = row["FlightDate"]

        # Compute UTC time directly using the UTC offset
        utc_offset = row["OriginTimezone"]  # UTC offset in hours (e.g., -8.0 for PST)
        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute
        )
        # Adjust the local time to UTC
        utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
        return utc_ts
    
    def arr_time_to_utc(row):
        hour = row["CRSArrTime"] // 100
        minute = row["CRSArrTime"] % 100

        # Account for overnight flights
        date = row["FlightDate"] + (
            pd.Timedelta(seconds=0) if row["CRSArrTime"] > row["CRSDepTime"] - 400 else pd.Timedelta(days=1)
        )

        # Compute UTC time directly using the UTC offset
        utc_offset = row["DestTimezone"]  # UTC offset in hours (e.g., -5.0 for EST)
        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute
        )
        # Adjust the local time to UTC
        utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
        return utc_ts

    def div1_arr_time_to_utc(row):
        hour = row["Div1WheelsOn"] // 100
        minute = row["Div1WheelsOn"] % 100

        date = row["FlightDate"]

        # Compute UTC time directly using the UTC offset
        utc_offset = row["DestTimezone"]  # UTC offset in hours (e.g., -5.0 for EST)
        local_ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute
        )
        # Adjust the local time to UTC
        utc_ts = local_ts - pd.Timedelta(hours=utc_offset)
        return utc_ts

    flights["CRSArrTime"] = flights["CRSArrTime"].fillna(0).astype(int)
    flights["CRSDepTime"] = flights["CRSDepTime"].fillna(0).astype(int)
    flights["Div1WheelsOn"] = flights["Div1WheelsOn"].fillna(0).astype(int)

    flights["ScheduledDepTimeUTC"] = flights.apply(dep_time_to_utc, axis=1)
    flights["ScheduledDepDateUTC"] = flights["ScheduledDepTimeUTC"].dt.date
    flights["ScheduledDepHourUTC"] = flights["ScheduledDepTimeUTC"].dt.hour
    # Pacific time conversion unnecessary
    # flights["ScheduledDepTimePacific"] = flights["ScheduledDepTimeUTC"].dt.tz_convert(
    #     "America/Los_Angeles"
    # )
    # flights["ScheduledDepDatePacific"] = flights["ScheduledDepTimePacific"].dt.date
    # flights["ScheduledDepHourPacific"] = flights["ScheduledDepTimePacific"].dt.hour

    flights["ScheduledArrTimeUTC"] = flights.apply(arr_time_to_utc, axis=1)
    flights["ScheduledArrDateUTC"] = flights["ScheduledArrTimeUTC"].dt.date
    flights["ScheduledArrHourUTC"] = flights["ScheduledArrTimeUTC"].dt.hour
    # flights["ScheduledArrTimePacific"] = flights["ScheduledArrTimeUTC"].dt.tz_convert(
    #     "America/Los_Angeles"
    # )
    # flights["ScheduledArrDatePacific"] = flights["ScheduledArrTimePacific"].dt.date
    # flights["ScheduledArrHourPacific"] = flights["ScheduledArrTimePacific"].dt.hour

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
