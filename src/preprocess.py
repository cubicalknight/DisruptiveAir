# copied from propagation
import duckdb
import pandas as pd
import timezonefinder as tzf

# TODO: Update directory paths
airport_coords = (
    pd.read_csv("../network_viz/T_MASTER_CORD.csv")
    .set_index("AIRPORT")
    .query("AIRPORT_IS_CLOSED < 1 and AIRPORT_IS_LATEST")[
        ["LATITUDE", "LONGITUDE", "DISPLAY_AIRPORT_NAME"]
    ]
)


# TODO: Update directory paths
# TODO: Change to user input dates
def prep_bts(start_date="2022-12-21", end_date="2022-12-30", table_name="flight_schedule"):
    with duckdb.connect("../transtats/duck.db") as db:
        flights = db.query(
            f"""
            select * from {table_name} mess
            left join (
                select id, list(odc_level) as levels from southwest_odcs group by id
            ) odcs on mess.column000 = odcs.id
            where mess.FlightDate between date '{start_date}' and date '{end_date}' and
            mess.column000 is not null"""
        ).to_df()
        finder = tzf.TimezoneFinder()

    flights["OriginTimezone"] = flights["Origin"].map(
        lambda origin: finder.timezone_at(
            lng=airport_coords.LONGITUDE[origin], lat=airport_coords.LATITUDE[origin]
        )
    )
    flights["DestTimezone"] = flights["Dest"].map(
        lambda origin: finder.timezone_at(
            lng=airport_coords.LONGITUDE[origin], lat=airport_coords.LATITUDE[origin]
        )
    )
    flights["Div1Timezone"] = flights["Div1Airport"].map(
        lambda div: None if div is None else finder.timezone_at(
            lng=airport_coords.LONGITUDE[div], lat=airport_coords.LATITUDE[div]
        )
    )

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
            tz=row["OriginTimezone"],
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
            tz=row["DestTimezone"],
        )
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

    flights["ScheduledDepTimeUTC"] = flights.apply(dep_time_to_utc, axis=1)
    flights["ScheduledDepDateUTC"] = flights["ScheduledDepTimeUTC"].dt.date
    flights["ScheduledDepHourUTC"] = flights["ScheduledDepTimeUTC"].dt.hour
    flights["ScheduledDepTimePacific"] = flights["ScheduledDepTimeUTC"].dt.tz_convert(
        "America/Los_Angeles"
    )
    flights["ScheduledDepDatePacific"] = flights["ScheduledDepTimePacific"].dt.date
    flights["ScheduledDepHourPacific"] = flights["ScheduledDepTimePacific"].dt.hour
    flights["ScheduledArrTimeUTC"] = flights.apply(arr_time_to_utc, axis=1)
    flights["ScheduledArrDateUTC"] = flights["ScheduledArrTimeUTC"].dt.date
    flights["ScheduledArrHourUTC"] = flights["ScheduledArrTimeUTC"].dt.hour
    flights["ScheduledArrTimePacific"] = flights["ScheduledArrTimeUTC"].dt.tz_convert(
        "America/Los_Angeles"
    )
    flights["ScheduledArrDatePacific"] = flights["ScheduledArrTimePacific"].dt.date
    flights["ScheduledArrHourPacific"] = flights["ScheduledArrTimePacific"].dt.hour

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
