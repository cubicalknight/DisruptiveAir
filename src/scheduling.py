"""Index and retrieve the planned flight schedule."""
import datetime
from preprocess import prep_bts
import pandas as pd

# TODO: Revise parameters based on updated BTS data
class Schedule:
    def __init__(self, date) -> None:
        self.events = prep_bts(start_date=date, end_date=date)[[
            'FlightDate',
            'Tail_Number',
            'Flight_Number_Reporting_Airline',
            'Origin',
            'Dest',
            'CRSDepTime',
            'DepTime',
            'DepDelay',
            'DepDelayMinutes',
            'DepDel15',
            'DepartureDelayGroups',
            'DepTimeBlk',
            'TaxiOut',
            'WheelsOff',
            'WheelsOn',
            'TaxiIn',
            'CRSArrTime',
            'ArrTime',
            'ArrDelay',
            'ArrDelayMinutes',
            'ArrDel15',
            'ArrivalDelayGroups',
            'Cancelled',
            'CancellationCode',
            'Diverted',
            'CRSElapsedTime',
            'ActualElapsedTime',
            'AirTime',
            'Flights',
            'Distance',
            'DistanceGroup',
            'CarrierDelay',
            'WeatherDelay',
            'NASDelay',
            'SecurityDelay',
            'LateAircraftDelay',
            'FirstDepTime',
            'TotalAddGTime',
            'LongestAddGTime',
            'DivAirportLandings',
            'DivReachedDest',
            'DivActualElapsedTime',
            'DivArrDelay',
            'DivDistance',
            # NOTE: diversions are not supported in this ABM
            # 'Div1Airport',
            # 'Div1AirportID',
            # 'Div1AirportSeqID',
            # 'Div1WheelsOn',
            # 'Div1TotalGTime',
            # 'Div1LongestGTime',
            # 'Div1WheelsOff',
            # 'Div1TailNum',
            # 'Div2Airport',
            # 'Div2AirportID',
            # 'Div2AirportSeqID',
            # 'Div2WheelsOn',
            # 'Div2TotalGTime',
            # 'Div2LongestGTime',
            # 'Div2WheelsOff',
            # 'Div2TailNum',
            # 'Div3Airport',
            # 'Div3AirportID',
            # 'Div3AirportSeqID',
            # 'Div3WheelsOn',
            # 'Div3TotalGTime',
            # 'Div3LongestGTime',
            # 'Div3WheelsOff',
            # 'Div3TailNum',
            # 'Div4Airport',
            # 'Div4AirportID',
            # 'Div4AirportSeqID',
            # 'Div4WheelsOn',
            # 'Div4TotalGTime',
            # 'Div4LongestGTime',
            # 'Div4WheelsOff',
            # 'Div4TailNum',
            # 'Div5Airport',
            # 'Div5AirportID',
            # 'Div5AirportSeqID',
            # 'Div5WheelsOn',
            # 'Div5TotalGTime',
            # 'Div5LongestGTime',
            # 'Div5WheelsOff',
            # 'Div5TailNum',
            'OriginTimezone',
            'OriginTimezoneName',
            'DestTimezone',
            'DestTimezoneName',
            'ScheduledDepTimeUTC',
            'ScheduledDepDateUTC',
            'ScheduledDepHourUTC',
            'ScheduledArrTimeUTC',
            'ScheduledArrDateUTC',
            'ScheduledArrHourUTC'
        ]]
        self.events["Revenue"] = True
        self.events = self.events[~self.events['Tail_Number'].isna()]
        self.event_by_aircraft = {
            num: table
            for num, table in self.events.
                sort_values(['FlightDate', 'ScheduledDepTimeUTC']).
                groupby('Tail_Number')
        }

    def all_tails(self) -> list[str]:
        return list(self.events['Tail_Number'].unique())
    
    def all_airports(self) -> list[str]:
        return list(pd.concat([self.events['Dest'], self.events['Origin']]).unique())

    def schedule(self, tail: str) -> pd.DataFrame:
        return self.event_by_aircraft[tail]

    def active_flight(self, model_time: datetime.datetime, schedule: pd.DataFrame) -> dict|None:
        # current ongoing, otherwise next up
        for _, row in schedule.iterrows():
            if row['ScheduledDepTimeUTC'].to_pydatetime() <= model_time <= row['ScheduledArrTimeUTC'].to_pydatetime():
                # ongoing flight
                return row
            if row['ScheduledDepTimeUTC'].to_pydatetime() > model_time:
                # next upcoming flight
                # schedule is ordered by time ascending, so this would be the next upcoming flight
                return row
        # the schedule is finished
        return None

    def model_start_time(self) -> datetime.datetime:
        return self.events['ScheduledDepTimeUTC'].min().to_pydatetime() - datetime.timedelta(minutes=-10)
    
    def airport_initial_crew_count(self, airport: str) -> int:
        """Generate a graph of crew-team counts at an airport."""
        day_flights = self.events[(self.events.Origin == airport) | (self.events.Dest == airport)]\
            .sort_values('ScheduledDepTimeUTC')
        events = sorted(list(day_flights.apply(lambda x: self.row_to_event(x, airport), axis=1)))
        counts = [0]
        for _time, change, _flight_number in events:
            counts.append(counts[-1] + change)
        # plt.show()
        return -min(counts)

    @staticmethod
    def row_to_event(row: pd.Series, airport: str):
        """Convert one BTS dataset row to an event in the form (time, crew change [+1/-1])"""
        arrival = row['Dest'] == airport
        if arrival:
            return (row['ScheduledArrTimeUTC'], 1, (row['Flight_Number_Reporting_Airline'],))
        else:
            return (row['ScheduledDepTimeUTC'], -1, (row['Flight_Number_Reporting_Airline'],))