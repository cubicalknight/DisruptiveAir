# %%
import abc
import json
import pathlib

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date, UTC
from scheduling import Schedule
from preprocess import airport_coords
import logging
import math
import random
from typing import Optional
import plotly.graph_objects as go
from dataclasses import dataclass
import click

VERSION = "1.1"


# %%
class Parameters:
    # minimum time between arriving at and departing from an airport
    min_turnaround = timedelta(minutes=35)
    # max delay before an aircraft agent gives up on a flight
    max_delay = timedelta(hours=6)
    # minimum time between flights for crew agents
    # TODO validate
    min_crew_turnaround = timedelta(minutes=30)


# %%
@dataclass
class CapacityConstraint:
    start_time: datetime
    end_time: datetime
    arrival_capacity: int|None = None
    departure_capacity: int|None = None

# %%
class Airport:
    clearance_rule = "ground_stop"
    crew_selection_rule = "random"

    code: str
    crews: list["Crew"]
    arrival_rate: dict[(date, int), int]
    arrival_capacity: int|None
    departure_rate: dict[(date, int), int]
    departure_capacity: int|None
    ground_stop: list[CapacityConstraint]

    metric_cancellations: int

    def __init__(self, code: str, network, dep_capacity: int|None = None, arr_capacity: int|None = None) -> None:
        self.code = code
        self.crews = []
        self.arrival_rate = {}
        self.departure_rate = {}
        self.metric_cancellations = 0
        self.ground_stop = []
        # per hour
        self.arrival_capacity = arr_capacity
        self.departure_capacity = dep_capacity
        network.all_airports[code] = self
        self.network = network

    def arrivals_ground_stopped(self, model_time: datetime) -> timedelta|None:
        """Checks the capacity constraints to see whether this airport allows clearances for flights to this airport to be issued."""
        for stop in self.ground_stop:
            if stop.start_time <= model_time <= stop.end_time and stop.arrival_capacity is not None:
                actual_rate = self.arrival_rate.get((model_time.date(), model_time.hour), 0)
                if actual_rate >= stop.arrival_capacity:
                    return timedelta(minutes=20) if stop.arrival_capacity > 0 else (stop.end_time - model_time)
        return None

    def departures_ground_stopped(self, model_time: datetime) -> bool:
        """Checks the capacity constraints to see whether this airport currently issues clearances for flights from it."""
        for stop in self.ground_stop:
            if stop.start_time <= model_time <= stop.end_time and stop.departure_capacity is not None:
                return self.departure_rate.get((model_time.date(), model_time.hour), 0) >= stop.departure_capacity
        return False

    def request_depart(self, flight, model_time: datetime) -> timedelta|None:
        dest: Airport = self.network.all_airports[flight['Dest']]
        match Airport.clearance_rule:
            case "none":
                return timedelta(seconds=0)
            case "ground_stop":
                ground_stop_delay = dest.arrivals_ground_stopped(model_time)
                if ground_stop_delay is not None:
                    # just delay the departure. let the aircraft itself decide when to cancel.
                    return ground_stop_delay
                if self.departures_ground_stopped(model_time):
                    return timedelta(minutes=20)
                return timedelta(seconds=0)
            case "actual":
                if flight['Cancelled']:
                    return None
                actual_deptime = flight['ScheduledDepTimeUTC'].to_pydatetime() + timedelta(minutes=flight['DepDelay'])
                return actual_deptime - model_time
            case "throughput":
                return timedelta(minutes=20) \
                    if self.departure_rate >= self.departure_capacity.get((model_time.date(), model_time.hour), 0) \
                    else timedelta(seconds=0)
            case _:
                raise ValueError(f"airport clearance rule {Airport.clearance_rule} not implemented")

    def request_arrive(self, flight, model_time: datetime) -> timedelta|None:
        match Airport.clearance_rule:
            case "none":
                return timedelta(seconds=0)
            case "ground_stop":
                # can't cancel arrival requests
                # diversion is not implemented
                # diversion is represented with a ridiculously long delay
                if self.arrivals_ground_stopped(model_time):
                    # just delay the arrival.
                    # TODO if the ground stop has a capacity limit of 0, delay until it is over.
                    return timedelta(minutes=20)
                return timedelta(seconds=0)
            case "actual":
                if math.isnan(flight['ArrDelay']):
                    return timedelta(days=5)
                actual_arr_time = flight['ScheduledArrTimeUTC'].to_pydatetime() + timedelta(minutes=flight['ArrDelay'])
                time_remaining = actual_arr_time - model_time
                if time_remaining <= timedelta(seconds=0):
                    return timedelta(seconds=0)
                return time_remaining
            case "throughput":
                return timedelta(minutes=10) \
                    if self.arrival_rate >= self.arrival_capacity.get((model_time.date(), model_time.hour), 0) \
                    else timedelta(seconds=0)
            case _:
                raise ValueError(f"airport clearance rule {Airport.clearance_rule} not implemented")

    def record_cancellation(self):
        """Add 1 to the counter of cancelled flights from this airport."""
        self.metric_cancellations += 1

    def find_crew_for(self, model_time: datetime, flight: pd.Series) -> Optional["Crew"]:
        """Find a crew agent at this airport to execute the given flight.
        
        If no crews are available, returns `None`."""
        able_crew = []
        for i, crew in enumerate(self.crews):
            hours_after = crew.hours_remaining_after(flight, model_time)
            if hours_after > 0 and crew.ready_for_next_flight(model_time):
                able_crew.append((i, crew, hours_after))

        if len(able_crew) == 0:
            return None
        match Airport.crew_selection_rule:
            case "random":
                return random.choice(able_crew)[1]
            case "greedy_utilization":
                # Find the crew for which the available flight time after this flight is maximized
                return max(able_crew, key=lambda x: x[2])[1]
            case _:
                raise NotImplementedError(Airport.crew_selection_rule)

    def depart(self, model_time: datetime, crew: "Crew"):
        key = (model_time.date(), model_time.hour)
        self.departure_rate.setdefault(key, 0)
        self.departure_rate[key] += 1
        assert crew in self.crews
        self.crews.remove(crew)

    def arrive(self, model_time: datetime, crew: "Crew"):
        key = (model_time.date(), model_time.hour)
        self.arrival_rate.setdefault(key, 0)
        self.arrival_rate[key] += 1
        assert crew not in self.crews
        self.crews.append(crew)

# %%
class Dispatcher:
    """Functionalities for attempting network recovery."""
    # greedy: iterate through all disrupted aircraft, sorted by disrupted time
    # for each aircraft, find the longest feasible path within the graph of unfulfilled flights, and claim that path
      # TODO consider preemptive risk
    recovery_strategy = 'none'
    unfulfilled_flights: list[pd.Series]

    metric_recoveries: int

    class RecoveryStrategy(abc.ABC):
        """A generic recovery strategy. All concrete strategies must inherit from this one."""
        @abc.abstractmethod
        def recover(self, flights: list[pd.Series], aircraft: list["Aircraft"], model_time: datetime) -> tuple[list["Aircraft"], list[pd.Series], list[pd.Series]]:
            """Recover the given `aircraft` by assigning the given unfulfilled `flights` to them.

            Returns:
             - The list of all aircraft from `aircraft` that were successfully recovered
            (i.e., they have a plan now).
             - The new list of unfulfilled flights.
             - List of unfulfilled flights to give up on."""
            raise NotImplementedError()

    class GreedyDfs(RecoveryStrategy):
        """Greedy Depth-First Search recovery strategy.

        Iterate through all the aircraft from earliest disrupted to latest disrupted.
        For each aircraft, search for and assign the longest path possible within unfulfilled flights.
        (In this version, do not consider deadheading a possibility. Deadheading means executing a non-revenue flight with crew only.)
        """
        flights: pd.DataFrame
        aircraft: list["Aircraft"]
        model_time: datetime

        def recover(self, flights: list[pd.Series], aircraft: list["Aircraft"], model_time: datetime) -> tuple[list["Aircraft"], list[pd.Series], list[pd.Series]]:
            if len(flights) == 0:
                # no flights to assign
                return [], flights, []

            self.flights = pd.DataFrame(flights)
            self.model_time = model_time
            self.aircraft: list["Aircraft"] = sorted(aircraft, key=lambda a: a.disrupted_time)
            give_up = self.flights[(model_time - self.flights['ScheduledDepTimeUTC']).dt.total_seconds() / 3600 > 6]
            print(f"solving recovery with {len(flights)} flights, {len(aircraft)} aircraft (giving up on {give_up.shape[0]} stale flights)")
            unfulfilled = set(self.flights.index).difference(set(give_up.index))
            recovered: list[Aircraft] = []

            for acft in self.aircraft:
                path = self.find_longest_path(acft.location, unfulfilled, model_time)
                if path is None:
                    continue
                acft.plan = pd.DataFrame(path)
                acft.flight_index = 0
                acft.disrupted_time = None
                recovered.append(acft)
                unfulfilled.difference_update(set(s.name for s in path))

            return recovered, [self.flights.loc[i] for i in unfulfilled], [r for _, r in give_up.iterrows()]

        def find_longest_path(self, start_airport: str, unfulfilled_set: set[int], execute_time: datetime, depth: int = 0) -> list[pd.Series]|None:
            if depth > 8: # prune the search to keep it tractable
                return None
            longest_path: list[pd.Series]|None = None
            longest_length = 0
            longest_deptime_sum = 0
            for _, succ in self.successors(start_airport, unfulfilled_set, execute_time).iterrows():
                unfulfilled_set.remove(succ.name)
                next_execute_time = max(
                    succ.ScheduledArrTimeUTC.to_pydatetime() + Parameters.min_turnaround,
                    execute_time + (succ.ScheduledArrTimeUTC.to_pydatetime() - succ.ScheduledDepTimeUTC.to_pydatetime()) + Parameters.min_turnaround
                )
                subpath = self.find_longest_path(succ.Dest, unfulfilled_set, next_execute_time, depth + 1)
                if subpath is None:
                    # This is a path of length 1
                    deptime_sum = (succ.ScheduledDepTimeUTC.to_pydatetime() - execute_time).total_seconds()
                    if longest_length == 0 or (longest_path == 1 and longest_deptime_sum > deptime_sum):
                        longest_path, longest_length, longest_deptime_sum = [succ], 1, deptime_sum
                else:
                    deptime_sum = sum((i.ScheduledDepTimeUTC.to_pydatetime() - execute_time).total_seconds() for i in [succ] + subpath)
                    if (longest_length < len(subpath) + 1) or (longest_length == len(subpath) + 1 and deptime_sum < longest_deptime_sum):
                        # We have found a longer path OR (our path is the max length so far AND it has a lower sum of departure times)
                        longest_path = [succ] + subpath
                        longest_length = len(subpath) + 1
                        longest_deptime_sum = deptime_sum
                unfulfilled_set.add(succ.name)
            return longest_path

        def successors(self, start: str, unfulfilled_set: set[int], execute_time: datetime) -> pd.DataFrame:
            """Find all flights that can feasibly be executed from the `unfulfilled_set` after finishing `finished_flight`.

            For the sake of realism, flights that have been delayed for over 6 hours at the estimated execution time
            cannot be picked up.
            """
            subflights = self.flights.loc[list(unfulfilled_set)]
            return subflights[
                (subflights.Origin == start) &
                (subflights.ScheduledDepTimeUTC.map(lambda d: execute_time - d.to_pydatetime() <= timedelta(hours=6)))]

    def __init__(self) -> None:
        self.unfulfilled_flights = []
        self.metric_recoveries = 0

    def unfulfill(self, flight: pd.Series):
        self.unfulfilled_flights.append(flight)

    def draw_problem_map(self, disrupted_aircraft: list["Aircraft"]) -> go.Figure:
        fig = go.Figure()
        for flight in self.unfulfilled_flights:
            origin_pos = airport_coords.loc[flight.Origin][['LATITUDE', 'LONGITUDE']]
            dest_pos = airport_coords.loc[flight.Dest][['LATITUDE', 'LONGITUDE']]
            fig.add_trace(go.Scattergeo(
                lon=[origin_pos.LONGITUDE, dest_pos.LONGITUDE],
                lat=[origin_pos.LATITUDE, dest_pos.LATITUDE],
                text=f'{flight.Flight_Number_Reporting_Airline} from {flight.Origin} to {flight.Dest}',
                mode='lines',
                line_color='red',
                line={
                    "color": "red",
                    "width": 2,
                }
            ))
        fig.add_trace(go.Scattergeo(
            lon=airport_coords.LONGITUDE.loc[[a.location for a in disrupted_aircraft]],
            lat=airport_coords.LATITUDE.loc[[a.location for a in disrupted_aircraft]],
            text=[str(a) for a in disrupted_aircraft],
            mode='markers',
            marker_color='blue',
        ))
        fig.update_layout(title=f'Southwest ABM Recovery Problem', geo_scope='usa')
        return fig

    def draw_problem_graph(self, disrupted_aircraft: list["Aircraft"], new_unfulfilled: list[pd.Series], recovered: list["Aircraft"]):
        """Draw a graph of the problem and the flights fulfilled by the current solution."""
        import networkx
        import matplotlib as mpl
        all_airports = list(set(f.Origin for f in self.unfulfilled_flights).union(f.Dest for f in self.unfulfilled_flights))
        dig = networkx.DiGraph()
        for f in self.unfulfilled_flights:
            dig.add_edge(all_airports.index(f.Origin), all_airports.index(f.Dest), color='black')

        # colors = ['g' if any(a.location == all_airports[loc] for a in disrupted_aircraft) else 'r' for loc in dig.nodes]
        # edge_colors = [dig[u][v]['color'] for u, v in dig.edges()]
        locations = {}
        for i, airport in enumerate(all_airports):
            locations[i] = (airport_coords.LONGITUDE.loc[airport], airport_coords.LATITUDE.loc[airport])

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        # Draw the problem
        networkx.draw_networkx(dig, ax=ax, pos=locations, node_size=40, with_labels=False, arrows=True)
        fig.savefig("recovery-problem-graph.pdf")

        # Draw the solution
        # We only render the first 10 recovered aircraft because our colormap doesn't have enough room
        solution = networkx.DiGraph()
        still_unfulfilled_names = set(u.name for u in new_unfulfilled)
        significant_airports = set()
        for f in self.unfulfilled_flights:
            # gray-dashed if not solved, colormap if in exactly one aircraft, bolded black if in multiple aircraft's plans
            aircrafts = ((i, ac) for i, ac in enumerate(recovered)
                             if any(f.Origin == plan.Origin and f.Dest == plan.Dest for _, plan in ac.plan.iterrows()))
            aircraft = next(aircrafts, None)
            if f.name in still_unfulfilled_names or aircraft is None:
                # not solved
                color = (0.8, 0.8, 0.8)
                style = 'dashed'
                weight = 2
            else:
                has_multiple = next(aircrafts, None) is not None
                color = mpl.colormaps['tab10'].colors[aircraft[0]] if not has_multiple else 'black'
                style = 'solid'
                weight = 4 if has_multiple else 2
                significant_airports.add(f.Origin)
                significant_airports.add(f.Dest)
            solution.add_edge(all_airports.index(f.Origin), all_airports.index(f.Dest),
                              color=color, style=style, weight=weight)
        def node_color_size(index: int) -> tuple[str, int]:
            aircrafts = ((i, ac) for i, ac in enumerate(recovered) if ac.location == all_airports[index])
            aircraft = next(aircrafts, None)
            if aircraft is not None:
                return (mpl.colormaps['tab10'].colors[aircraft[0]], 40)
            elif all_airports[index] in significant_airports:
                return ('black', 25)
            else:
                return ((0.8, 0.8, 0.8), 15)
        
        fig.clf()
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        colors_sizes = [node_color_size(loc) for loc in solution.nodes]
        networkx.draw_networkx(solution, ax=ax, pos=locations, node_size=[t[1] for t in colors_sizes], node_color=[t[0] for t in colors_sizes], with_labels=False,
                               edge_color=[solution[u][v]['color'] for u, v in solution.edges()],
                               style=[solution[u][v]['style'] for u, v in solution.edges()])
        networkx.draw_networkx_labels(solution,
                                      labels=dict((i, a) for i, a in enumerate(all_airports) if a in significant_airports),
                                      pos={i: (x, y - 0.9) for i, (x, y) in locations.items()},
                                      alpha=0.8)
        fig.tight_layout()
        fig.savefig("recovery-solution-graph.pdf")

        print('WROTE PROBLEM SAMPLE')

    def create_recovery_solver(self) -> RecoveryStrategy|None:
        match Dispatcher.recovery_strategy:
            case "none":
                return None
            case "greedy":
                return Dispatcher.GreedyDfs()
            case _:
                raise NotImplementedError(f"Recovery strategy {Dispatcher.recovery_strategy} not implemented")

    def recover(self, model: "Network", model_time: datetime, disrupted_aircraft: list["Aircraft"]) -> list["Aircraft"]:
        """Try to recover the given list of disrupted aircraft by assigning them to unfulfilled flights.

        Returns: The list of aircraft successfully recovered (and thus should take action soon)"""
        # strategies: no dispatching
        #   greedy individual matching

        solver = self.create_recovery_solver()
        if solver is None:
            # cancel all unfulfilled flights
            for flight in self.unfulfilled_flights:
                model.all_airports[flight.Origin].record_cancellation()
            self.unfulfilled_flights.clear()
            return []

        recovered_aircraft, new_unfulfilled, give_up = solver.recover(self.unfulfilled_flights, disrupted_aircraft, model_time)
        self.metric_recoveries += len(recovered_aircraft)

        # if 45 < len(self.unfulfilled_flights) < 100 and len(new_unfulfilled) < len(self.unfulfilled_flights) - 2:
        #     for flight in sorted(self.unfulfilled_flights, key=lambda f: f.ScheduledDepTimeUTC):
        #         print(f'(originally {flight.Tail_Number}) from {flight.Origin} to {flight.Dest} ({flight.CRSDepTime}) [{(flight.ScheduledDepTimeUTC - model_time).total_seconds() / 3600} hours until departure]')
        #     print(recovered_aircraft)
        #     print([s.plan for s in recovered_aircraft])
        #     self.draw_problem_graph(disrupted_aircraft, new_unfulfilled, recovered_aircraft)

        self.unfulfilled_flights = new_unfulfilled
        for flight in give_up:
            model.all_airports[flight['Origin']].record_cancellation()
        return recovered_aircraft

# %%

from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class Flight:
    """A singular flight from one origin to one destination.
    
    Each Flight must have been flown in the model.
    Its parameters are set according to how the flight was flown in the model, not according to any schedule.
    Hence, it is necessary in addition to the DataFrames."""
    origin: str
    dest: str
    aircraft: "Aircraft"
    pilots: "Crew"
    sched_depart: datetime
    sched_arrive: datetime
    depart_time: datetime
    arrive_time: datetime
    plan: Optional[pd.Series]

class Aircraft:
    tail: str # like N945WN
    location: str|None # like DAL
    crew: Optional["Crew"]
    # if location is None, then this aircraft is in the air
    current_phase_start_time: datetime|None
    cleared_transition_time: datetime|None
    flight_index: int|None
    plan: pd.DataFrame
    disrupted_time: datetime|None
    diverted: bool
    log: logging.Logger
    network: "Network"
    flight_history: list[Flight]

    metrics_flights_completed: int
    metrics_revenue_flights_completed: int

    _memo_current_delay: float|None

    def __init__(self, network, tail: str, logfile=None, loglevel="DEBUG") -> None:
        self.network = network
        self.tail = tail
        self.plan = network.schedule.schedule(tail)
        self.location = self.plan['Origin'].iloc[0]
        self.current_phase_start_time = None
        self.cleared_transition_time = None
        self.flight_index = 0
        self.disrupted_time = None
        self.diverted = False
        self.crew = None
        self.log = logging.getLogger(tail)
        self.log.setLevel(loglevel)
        if len(self.log.handlers) == 0:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(name)s %(levelname)8s: %(message)s"
            )
            handler.setFormatter(fmt)
            if logfile is not None:
                file_handler = logging.FileHandler(logfile)
                file_handler.setFormatter(fmt)
                self.log.addHandler(file_handler)
            self.log.addHandler(handler)
        self.metrics_flights_completed = 0
        self.metrics_revenue_flights_completed = 0
        self.flight_history = []

        self._memo_current_delay = None

    @property
    def flight(self):
        if self.flight_index is None or self.flight_index >= self.plan.shape[0]:
            return None
        return self.plan.iloc[self.flight_index]

    def step(self, model_time: datetime) -> Optional[datetime]:
        """Process a model time step for this agent.

        If airborne and flight time is reached, obtain clearance to land. If clearance granted, transition to the ground state.
        If on the ground and min(turnaround time, scheduled departure time) is reached, obtain clearance to depart. If clearance granted, transition to the airborne state.
            If flight is cancelled, mark this aircraft as disrupted for the dispatcher to systematically recover.

        Returns:
        The next model time where this function should be called again.
        If `None`, do not call this function again."""

        # transition: flying -> ground
        # transition is only possible when:
        #   1. the requisite time airborne has elapsed
        #   2. the landing clearance is given by the Dispatcher agent
        if self.airborne:
            assert self.flight is not None
            flight_duration = self.flight['ScheduledArrTimeUTC'].to_pydatetime() - self.flight['ScheduledDepTimeUTC'].to_pydatetime()
            dest_reached = (model_time - self.current_phase_start_time) >= flight_duration
            if not dest_reached:
                # flight in progress
                return self.flight['ScheduledArrTimeUTC'].to_pydatetime()
            if self.cleared_transition_time is not None and self.cleared_transition_time > model_time:
                # waiting for delay to expire
                return self.cleared_transition_time
            clearance = self.network.all_airports[self.flight['Dest']].request_arrive(self.flight, model_time)
            assert clearance is not None, "can't cancel landing clearance"
            if clearance <= timedelta(seconds=1):
                # transition approved, so make it now
                self.arrive_at(self.flight['Dest'], model_time)
                if self.flight is None:
                    # TODO enqueue aircraft again if flights are populated
                    return None
                return (self.flight['ScheduledDepTimeUTC'] \
                    if self.current_phase_start_time is None else max(
                        self.current_phase_start_time + Parameters.min_turnaround,
                        self.flight['ScheduledDepTimeUTC']
                    ))
            else:
                # transition delayed
                if clearance > timedelta(days=1):
                    # diverted (unavailable)
                    self.diverted = True
                    self.log.warning("Flight from %s to %s DIVERTED (no longer available)", self.flight['Origin'], self.flight['Dest'])
                    return None
                self.log.info("Arrival at %s delayed by %s", self.flight['Dest'], str(clearance))
                self.cleared_transition_time = model_time + clearance
                return self.cleared_transition_time

        # transition: ground -> flying
        # transition is only possible when:
        #   1. the scheduled departure time has been reached
        #   2. the minimum turnaround time has elapsed since landing
        #   3. the departure clearance is given by the Dispatcher agent
        #   4. there is a crew agent at the origin that can pilot this flight
        elif self.flight is not None:
            if self.flight['Origin'] != self.location:
                if not self.disrupted_time:
                    self.log.warning("Location mismatch: planned flight departs from %s, but I am at %s", self.flight['Origin'], self.location)
                    self._cancel(model_time)
                return None # dispatcher should enqueue this aircraft's next update
            latest_departure_time = self.flight['ScheduledDepTimeUTC'] + Parameters.max_delay
            earliest_departure_time = self.flight['ScheduledDepTimeUTC'] \
                if self.current_phase_start_time is None else max(
                    self.current_phase_start_time + Parameters.min_turnaround,
                    self.flight['ScheduledDepTimeUTC']
                )
            if earliest_departure_time > latest_departure_time:
                self._cancel(model_time)
                return None
            if model_time < earliest_departure_time:
                # turnaround in progress / waiting for passengers
                return earliest_departure_time
            if self.cleared_transition_time is not None and model_time < self.cleared_transition_time:
                # waiting for delay to expire
                if self.cleared_transition_time > latest_departure_time and self.flight is not None:
                    self._cancel(model_time)
                    return None
                return self.cleared_transition_time
            crew = self.network.all_airports[self.location].find_crew_for(model_time, self.flight)
            if crew is None:
                # waiting for crew to become available
                # TODO aggregate minutes delayed by reason
                self.log.info("Departure from %s to %s delayed due to lack of crew",
                              self.flight["Origin"], self.flight["Dest"])
                if model_time + timedelta(minutes=10) >= latest_departure_time:
                    # timeout
                    self.log.info("Departure from %s to %s delayed beyond threshold due to lack of crew. Cancelled",
                                  self.flight['Origin'], self.flight['Dest'])
                    self._cancel(model_time)
                    return None
                return model_time + timedelta(minutes=10)
            if model_time > latest_departure_time:
                self._cancel(model_time)
                return None
            clearance = self.network.all_airports[self.location].request_depart(self.flight, model_time)
            if clearance is None:
                self.log.info("Departure clearance for %s (%s->%s) cancelled",
                              self.flight['Flight_Number_Reporting_Airline'],
                              self.flight['Origin'], self.flight['Dest'])
                self._cancel(model_time)
                return None
            elif clearance <= timedelta(seconds=1):
                # immediately cleared to the destination
                self.depart(model_time, crew)
                # minimum flight duration is the scheduled duration
                # TODO think about stochastically varying the actual flight time required
                flight_time = self.flight['ScheduledArrTimeUTC'].to_pydatetime() - \
                        self.flight['ScheduledDepTimeUTC'].to_pydatetime()
                self.log.debug("Next update is in %d minutes", flight_time.total_seconds()/60)
                return model_time + flight_time
            else:
                # delayed
                self.log.info("Departure from %s to %s delayed by %s",
                              self.flight['Origin'], self.flight['Dest'], str(clearance))
                self.cleared_transition_time = model_time + clearance
                if self.cleared_transition_time >= latest_departure_time:
                    # reached max delay threshold.
                    # at this point, this aircraft should give up trying to execute this flight.
                    self.log.info("Departure from %s to %s delayed beyond threshold. Cancelled",
                                  self.flight['Origin'], self.flight['Dest'])
                    self._cancel(model_time)
                    return None
                return self.cleared_transition_time

        else:
            # idling
            self.log.info("step() idling")
            return None

    def _cancel(self, model_time: datetime):
        assert self.flight is not None
        # cancelled. disrupted state
        # TODO think about preemptive cancellation.
        #   if we have a predicted ground stop, then preepmtively place a future flight on the "reserve list"
        #   this is true of several disruption schemes, including BTS replication.
        self.disrupted_time = model_time
        for flight in self.remaining_flights():
            assert id(flight) != id(self.flight), "Don't put the cancelled flight in the list of unfulfilled flights"
            self.network.dispatcher.unfulfill(flight)
        self.network.all_airports[self.location].record_cancellation()
        self.flight_index = None
        # it is always the dispatcher's job to recover aircraft.
        # the dispatcher is expected to change aircraft.plan if aircraft.disrupted_time is not None
        # then, it should reset flight_index and disrupted.
        self._on_phase_change()

    def arrive_at(self, dest: str, model_time: datetime):
        self.log.info("Arriving at %s from %s with crew %d (scheduled=%s actual=%s offset=%s)",
                      dest, self.flight['Origin'], self.crew.id,
                      self.flight['ScheduledArrTimeUTC'].to_pydatetime().time(),
                      model_time.time(),
                      (model_time - self.flight['ScheduledArrTimeUTC'].to_pydatetime()).total_seconds()//60)
        self.metrics_flights_completed += 1
        if self.flight['Revenue']:
            self.metrics_revenue_flights_completed += 1
        self.location = dest
        flight = Flight(
            origin=self.flight['Origin'],
            dest=dest,
            aircraft=self,
            pilots=self.crew,
            sched_depart=self.flight['ScheduledDepTimeUTC'].to_pydatetime(),
            sched_arrive=self.flight['ScheduledArrTimeUTC'].to_pydatetime(),
            depart_time=self.current_phase_start_time,
            arrive_time=model_time,
            plan=self.flight)
        self.flight_history.append(flight)
        self.current_phase_start_time = model_time
        self.cleared_transition_time = None
        self.flight_index += 1

        self.network.all_airports[dest].arrive(model_time, self.crew)
        self.crew.complete(flight)
        self.crew = None

        self._on_phase_change()

    def depart(self, model_time: datetime, crew: "Crew"):
        # assumes that self.flight is up-to-date
        assert self.flight is not None, "self.depart() when self.flight is None"
        self.log.info("Departing from %s for %s with crew %d (scheduled=%s actual=%s offset=%s)",
                      self.flight.Origin, self.flight.Dest, crew.id,
                      self.flight['ScheduledDepTimeUTC'].to_pydatetime().time(), model_time.time(),
                      (model_time - self.flight['ScheduledDepTimeUTC'].to_pydatetime()).total_seconds()//60)
        self.network.all_airports[self.location].depart(model_time, crew)
        crew.take_flight(self.tail)
        self.crew = crew
        self.location = None
        self.current_phase_start_time = model_time
        self.cleared_transition_time = None

        self._on_phase_change()

    def current_delay(self) -> float|None:
        """Retrieve the current minutes delayed for this aircraft.

        When this aircraft is on the ground (not self.airborne), its delay is based on its earliest next departure time.
        When this aircraft is airborne, its delay is based on its last departure time."""
        if self._memo_current_delay is not None:
            # assert self._memo_current_delay == self._compute_current_delay()
            return self._memo_current_delay
        else:
            return self._compute_current_delay()

    def _compute_current_delay(self) -> float|None:
        if self.airborne:
            return (self.current_phase_start_time - self.flight['ScheduledDepTimeUTC'].to_pydatetime()).total_seconds() / 60

        if self.flight is None:
            return None

        # TODO deduplicate
        earliest_departure_time = self.flight['ScheduledDepTimeUTC'] \
            if self.current_phase_start_time is None else max(
                self.current_phase_start_time + Parameters.min_turnaround,
                self.flight['ScheduledDepTimeUTC']
            )
        return (earliest_departure_time - self.flight['ScheduledDepTimeUTC'].to_pydatetime()).total_seconds() / 60

    def effective_location(self, model_time: datetime):
        """Deduce the most likely location of this agent in the world in lat/long coordinates.

        When this aircraft is on the ground (not self.airborne), it is located at self.location.
        When this aircraft is airborne, its location is linearly interpolated between the origin and destination."""
        if not self.airborne:
            return airport_coords.loc[self.location][['LATITUDE', 'LONGITUDE']]
        origin_loc = airport_coords.loc[self.flight['Origin']][['LATITUDE', 'LONGITUDE']]
        dest_loc = airport_coords.loc[self.flight['Dest']][['LATITUDE', 'LONGITUDE']]
        flight_duration = self.flight['ScheduledArrTimeUTC'].to_pydatetime() - self.flight['ScheduledDepTimeUTC'].to_pydatetime()
        proportion_done = (model_time - self.current_phase_start_time) / flight_duration
        proportion_done = max(0, min(1, proportion_done))
        # simple approximation via linear interpolation
        return dest_loc * proportion_done + origin_loc * (1 - proportion_done)

    def remaining_flights(self) -> list[pd.Series]:
        index = (self.flight_index + 1) if self.flight_index is not None else None
        return [series for _, series in self.plan.iloc[index:, :].iterrows()]

    @property
    def airborne(self) -> bool:
        return self.location is None

    @property
    def finished(self) -> bool:
        return self.flight_index is None or self.flight_index >= self.plan.shape[0]

    def __repr__(self) -> str:
        if self.airborne:
            return f"<{self.tail} flying {self.flight['Flight_Number_Reporting_Airline']} from {self.flight['Origin']} to {self.flight['Dest']}>"
        return f"<{self.tail} on the ground at {self.location}>"

    def _on_phase_change(self) -> None:
        """React to a change in phase (ground <-> airborne) by updating memoized metrics."""
        self._memo_current_delay = self._compute_current_delay()

# %%
from dataclasses import dataclass, field

@dataclass(order=True)
class NextUpdate:
    priority: datetime
    neg_delay_minutes: float
    item: Aircraft=field(compare=False)

# %%
from queue import PriorityQueue
from collections import deque

def negate_optional(delay: float|None) -> float:
    return -delay if delay is not None else None

# TODO add crew agents
# for simplification, start with only pilots
# CFR: ≤10 hours per 24-hour window for pilots working with co-pilots

# Each crew agent has an ID. *Simplifying assumption: Each crew agent actually represents a two-member crew team.*
# When a flight needs to depart, it needs to pick crew agents that will not timeout in the middle of the flight.
#   TODO determine heuristic to use for which crew agents to pick.
# RQ: What is the distribution of the overall duration of each flight string identified by flight number?
#   Answer: Skewed to the right, ranging from 100 to 800 minutes. Mostly 100-300 minutes.

class Crew:
    id: int
    # sorted by actual departure time
    flight_history: list[Flight]
    # future: deadhead_history
    # if in_flight, then aircraft tail number; otherwise, airport code
    current: str
    in_flight: bool

    def __init__(self, id: int, starting_airport: str) -> None:
        self.id = id
        self.flight_history = []
        self.current = starting_airport
        self.in_flight = False

    def take_flight(self, tail):
        self.current = tail
        self.in_flight = True
    
    def complete(self, flight: Flight):
        self.flight_history.append(flight)
        self.current = flight.dest
        self.in_flight = False

    def hours_remaining_after(self, next_flight: pd.Series, model_time: datetime) -> float:
        """Checks whether this crew agent can take on the given flight.
        
        Computes the workload of this crew with respect to the 10-hour limit."""

        # QUESTION: How do crew members "time out" in meltdowns?
        # ANSWER: When held on the ground but after pushing back due to an EDCT change or emerging weather system
        # formula: did we exceed 10-x hours of flight time in the past 24-x hours, where x is the next flight's duration?
        # "past" counts from the next flight's scheduled departure time or model_time, whichever is later
        flight_duration = next_flight['ScheduledArrTimeUTC'].to_pydatetime() - next_flight['ScheduledDepTimeUTC'].to_pydatetime()
        est_departure_time = max(next_flight['ScheduledDepTimeUTC'].to_pydatetime(), model_time)
        window_overlap_start = est_departure_time - timedelta(hours=24) + flight_duration
        window_flight_time = self._flight_time_during(window_overlap_start, est_departure_time)
        # is 9.9 to adjust for unexpected delays on arrival
        return 9.9 - window_flight_time - flight_duration.total_seconds()/3600
    
    def flight_time_in_last_24(self, model_time: datetime):
        return self._flight_time_during(model_time - timedelta(hours=24), model_time)
    
    def ready_for_next_flight(self, model_time: datetime):
        return len(self.flight_history) == 0 or \
            model_time - self.flight_history[-1].arrive_time >= Parameters.min_crew_turnaround

    def _flight_time_during(self, start: datetime, end: datetime) -> float:
        """Counts the number of hours of flight time in the given interval according to history."""
        # first, traverse until the flight departs before end
        idx = len(self.flight_history) - 1
        while idx >= 0 and self.flight_history[idx].depart_time >= end:
            idx -= 1
        flight_time_sum = 0
        # then, traverse until the flight departs before start
        while idx >= 0 and self.flight_history[idx].depart_time >= start:
            flight_time_sum += \
                (self.flight_history[idx].arrive_time - self.flight_history[idx].depart_time).total_seconds() / 3600
            idx -= 1
        # check if start occurs during the flight
        # if so, put the part of the flight inside the interval into the sum
        if idx >= 0 and self.flight_history[idx].arrive_time >= start:
            flight_time_sum += \
                (self.flight_history[idx].arrive_time - start).total_seconds() / 3600
        return flight_time_sum
    
    def __eq__(self, other):
        return isinstance(other, Crew) and other.id == self.id


class Network:
    aircraft: list[Aircraft]
    crews: list[Crew]
    schedule: Schedule
    dispatcher: Dispatcher
    log: logging.Logger
    metrics: deque[pd.Series]
    updates: PriorityQueue[NextUpdate]
    airport_tracked: list[str]
    all_airports: dict[str, Airport]

    report_dir: pathlib.Path
    report_log: pathlib.Path
    report_metrics: pathlib.Path
    report_throughput: pathlib.Path

    def __init__(self, date: str, report_name: str, loglevel="DEBUG", initial_crew_mult=1.0) -> None:
        self.aircraft = []
        self.crews = []
        self.schedule = Schedule(date)
        self.dispatcher = Dispatcher()
        self.updates = PriorityQueue()
        self.metrics = deque()
        self.all_airports = {}
        self.airports_tracked = []
        self._setup_report(report_name)
        self._setup_log(loglevel)

        for arpt in self.schedule.all_airports():
            self.all_airports[arpt] = Airport(arpt, self)
            for _ in range(int(initial_crew_mult * self.schedule.airport_initial_crew_count(arpt))):
                c = Crew(len(self.crews) + 1, arpt)
                self.crews.append(c)
                self.all_airports[arpt].crews.append(c)

        self.time = self.schedule.model_start_time()
        for tail in self.schedule.all_tails():
            aircraft = Aircraft(self, tail, logfile=str(self.report_log), loglevel=loglevel)
            self.aircraft.append(aircraft)
            self.updates.put(NextUpdate(self.time, negate_optional(aircraft.current_delay()), aircraft))


    def _setup_report(self, report_name: str) -> None:
        self.report_dir = pathlib.Path("reports") / report_name
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report_log = self.report_dir / "network.log"
        self.report_metrics = self.report_dir / "metrics.csv"
        self.report_throughput = self.report_dir / "throughput.csv"
        self.report_crew = self.report_dir / "final_crew.csv"
        self.report_flights = self.report_dir / "flights.csv"

    def _setup_log(self, loglevel: str) -> None:
        self.log = logging.getLogger("Network")
        with self.report_log.open(mode='w+') as f:
            f.write('')
        if len(self.log.handlers) == 0:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(name)s %(levelname)9s: %(message)s"
            )
            handler.setFormatter(fmt)
            file_handler = logging.FileHandler(self.report_log)
            file_handler.setFormatter(fmt)
            self.log.addHandler(file_handler)
            self.log.setLevel(loglevel)
            self.log.addHandler(handler)

    def step(self) -> datetime:
        """Execute one timestep in the entire network.

        Returns:
        The system timestep where the next step() call should take place."""

        self.log.info(" --- NEW TIME STEP %s --- ", str(self.time))
        while True:
            with self.updates.mutex:
                # Is the next update later than the current system timestamp?
                if len(self.updates.queue) == 0:
                    return None
                if self.updates.queue[0].priority > self.time:
                    return self.updates.queue[0].priority
            aircraft = self.updates.get().item
            next_update = aircraft.step(self.time)
            self.updates.task_done()
            if next_update is not None:
                update = NextUpdate(next_update, negate_optional(aircraft.current_delay()), aircraft)
                self.updates.put(update)

    def recover(self):
        """Run the dispatcher's recovery mechanism, and put recovered aircraft back in the updates PQ."""
        disrupted = [a for a in self.aircraft if a.disrupted_time is not None]
        recovered = self.dispatcher.recover(self, self.time, disrupted)
        for rec in recovered:
            self.updates.put(NextUpdate(self.time + timedelta(minutes=1), negate_optional(rec.current_delay()), rec))

    def scattergeo(self):
        """Create a geographic plot showing all the aircraft's locations and state."""
        positions = [aircraft.effective_location(self.time) for aircraft in self.aircraft]
        delays = [a.current_delay() for a in self.aircraft]
        return go.Scattergeo(
            lon=[pos['LONGITUDE'] for pos in positions],
            lat=[pos['LATITUDE'] for pos in positions],
            text=[str(a) for a in self.aircraft],
            mode='markers',
            marker_color=[d if d is not None else 0 for d in delays],
            marker={
                "size": 6,
                "colorscale": 'bluered',
                'cmin': 0,
                'cmax': 60,
                "colorbar": {
                    "title": "Delay w.r.t. Schedule"
                }
            }
        )

    def airports_scattergeo(self) -> go.Scattergeo:
        """Create a scatter trace that labels all the airports involved."""
        airports = list(self.all_airports.keys())
        return go.Scattergeo(
            lon=airport_coords['LONGITUDE'].loc[airports],
            lat=airport_coords['LATITUDE'].loc[airports],
            text=airports,
            textposition='top center',
            mode='markers+text',
            legend=None,
            textfont = {"color": 'black',
                    "family":'Helvetica Neue',
                    "size":9},
            marker={
                "size": 8,
                "color": "black",
                "line_color": "black",
            }
        )

    def make_figure(self) -> go.Figure:
        """Create a full-fledged figure visualizing the current state of the network."""
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2,
                            specs=[[{"colspan": 2, "type": "scattergeo"}, None], [{"type": "scatter"}, {"type": "scatter"}]],
                            row_heights=[0.7, 0.3],
                            subplot_titles=("Network", "Cancellation count", "Unfulfilled flight count"))
        fig.add_trace(self.scattergeo(), row=1, col=1)
        fig.add_trace(self.airports_scattergeo(), row=1, col=1)
        df = pd.DataFrame(self.metrics)
        fig.add_trace(go.Scatter(x=df['ModelHour'], y=df['TotalCancelled']), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['ModelHour'], y=df['UnfulfilledFlights']), row=2, col=2)
        fig.update_layout(
            title=f'Southwest ABM Aircraft (system time = {self.time} UTC)<br>'
                    f"Disruption: {Airport.clearance_rule}, Recovery: {Dispatcher.recovery_strategy}",
            geo_scope='usa',
            showlegend=False)
        return fig

    def log_stats(self):
        num_disrupted = sum([1 for a in self.aircraft if a.disrupted_time is not None])
        self.log.info("STEP STATS: %d aircraft disrupted", num_disrupted)

    def track_airports(self, airports: list[str]):
        self.airports_tracked = airports

    def record_metrics(self):
        """Record metrics of interest by appending them to self.metrics."""
        row = pd.Series({
            'Time': self.time,
            'TotalSystemDelay': 0,
            'DelayedAircraft': 0,
            'DisruptedAircraft': 0,
            'TotalAircraft': len(self.aircraft),
            'AircraftAirborne': 0,
            'AircraftFinished': 0,
            'TotalFlights': 0,
            'ModelHour': (self.time - self.schedule.model_start_time()).total_seconds() / 3600,
            'TotalRevenueFlights': 0,
            'TotalCancelled': 0,
            'CrewsInFlight': 0,
            'UnfulfilledFlights': len(self.dispatcher.unfulfilled_flights)
        })
        for aircraft in self.aircraft:
            if aircraft.disrupted_time is not None:
                row['DisruptedAircraft'] += 1
            if aircraft.current_delay() is not None:
                if aircraft.current_delay() >= 30:
                    row['DelayedAircraft'] += 1
                row['TotalSystemDelay'] += aircraft.current_delay()
            if not aircraft.diverted and aircraft.airborne:
                row['AircraftAirborne'] += 1
            if not aircraft.diverted and aircraft.finished:
                row['AircraftFinished'] += 1
            row['TotalFlights'] += aircraft.metrics_flights_completed
            row['TotalRevenueFlights'] += aircraft.metrics_revenue_flights_completed

        for airport_code in self.airports_tracked:
            row[f'{airport_code}.CancelledDepartures'] = self.all_airports[airport_code].metric_cancellations

        for airport in self.all_airports.values():
            row['TotalCancelled'] += airport.metric_cancellations
        
        for crew in self.crews:
            if crew.in_flight:
                row['CrewsInFlight'] += 1

        self.metrics.append(row)

    def crew_utilization(self):
        return pd.DataFrame([
            {
                'ID': crew.id,
                'Location': crew.current,
                'Airborne': crew.in_flight,
                'FlightTime': crew.flight_time_in_last_24(crew.flight_history[-1].arrive_time)
                    if len(crew.flight_history) > 0 else 0.0,
                'History': ' + '.join(f'{f.aircraft.tail}[{f.origin}-{f.dest}]'for f in crew.flight_history)
            } for crew in self.crews
        ])

    def airport_rates(self, airports: list[str]) -> pd.DataFrame:
        df = pd.DataFrame()
        for airport in airports:
            airport_obj = self.all_airports[airport]
            date_hour_keys = set(airport_obj.departure_rate.keys()).union(set(airport_obj.arrival_rate.keys()))
            adf = pd.DataFrame(
                [(date, hour, airport_obj.departure_rate.get((date, hour), 0), airport_obj.arrival_rate.get((date, hour), 0)) for (date, hour) in date_hour_keys],
                columns=['Date', 'Hour', 'DepCount', 'ArrCount'])
            adf['Airport'] = airport
            df = pd.concat([df, adf])
        return df

    def write_report(self) -> None:
        pd.DataFrame(self.metrics).to_csv(self.report_metrics)
        self.airport_rates(self.airports_tracked).to_csv(self.report_throughput)
        self.crew_utilization().to_csv(self.report_crew, index=False)
        self.flown_flights().to_csv(self.report_flights, index=False)

    def flown_flights(self) -> pd.DataFrame:
        flights = [
            {
                'tail': acft.tail,
                'crew': flt.pilots.id,
                'sched_depart': flt.sched_depart,
                'sched_arrive': flt.sched_arrive,
                'depart': flt.depart_time,
                'arrive': flt.arrive_time,
                'origin': flt.origin,
                'dest': flt.dest
            }
            for acft in self.aircraft for flt in acft.flight_history
        ]
        return pd.DataFrame(flights)
        
# Example ground stop scheme for DEN from Winter Storm Elliot
# TODO: Dynamically adapt based on outputs from learned weather outputs
# TODO: Note shift to LGA
ground_stop_schemes = {
    'DEN-Beginning': {'DEN': [
        CapacityConstraint(
            start_time=datetime(2022, 12, 22, 10, 0, tzinfo=UTC),
            end_time=datetime(2022, 12, 23, 0, 0, tzinfo=UTC),
            arrival_capacity=0,
            departure_capacity=5
        ),
        CapacityConstraint(
            start_time=datetime(2022, 12, 23, 0, 0, tzinfo=UTC),
            end_time=datetime(2022, 12, 23, 12, 0, tzinfo=UTC),
            arrival_capacity=5,
            departure_capacity=3
        ),
        CapacityConstraint(
            start_time=datetime(2022, 12, 23, 12, 0, tzinfo=UTC),
            end_time=datetime(2022, 12, 23, 23, 0, tzinfo=UTC),
            arrival_capacity=15,
            departure_capacity=10
        )
    ]}
}

@click.command()
@click.option('--visualize', '-p', help='Export frame images from the simulation', is_flag=True, default=False)
@click.option('--crew-selection', '-c', help='The crew selection strategy', type=click.Choice(['random', 'greedy_utilization']), default='random')
@click.option('--disruption', '-i', required=True, help='The disruption model', type=click.Choice(['none', 'ground_stop', 'actual']))
@click.option('--ground-stop-scheme', '-g', required=False, default='DEN-Beginning')
@click.option('--track', '-t', help='The list of airports to track the number of cancellations and export the traffic rates', required=False)
@click.option('--recovery', '-r', required=True, help='The recovery model', type=click.Choice(['none', 'greedy', 'lp', 'annealing']))
@click.option('--date', '-d', required=True, help='The date to simulate')
@click.option('--crew-multiplier', required=False, help='The multiplier applied to the initial crew count at all airports', type=float, default=1.0)
@click.option('--max-delay', required=False, help='The maximum minutes of delay an uncancelled flight may have. Exceeding this number of minutes automatically cancels the flight.', type=int, default=360)
@click.option('--verbose', '-v', count=True, default=0)
def run_model(visualize, crew_selection, disruption, ground_stop_scheme, track, recovery, date, crew_multiplier, max_delay, verbose):
    """Run the entire model according to the given options."""
    config = f"{date}-{crew_selection}-{disruption}-{recovery}-{crew_multiplier}-{max_delay}_{VERSION}"
    Airport.clearance_rule = disruption
    Airport.crew_selection_rule = crew_selection
    Dispatcher.recovery_strategy = recovery
    Parameters.max_delay = timedelta(minutes=max_delay)
    net = Network(date, report_name=config, loglevel=["WARN", "INFO", "DEBUG"][verbose],
                  initial_crew_mult=crew_multiplier)
    if track:
        net.track_airports(track.split(','))

    if disruption == "ground_stop":
        assert ground_stop_scheme in ground_stop_schemes, f"Unknown scheme: {ground_stop_scheme}"
        for airport, constraint in ground_stop_schemes[ground_stop_scheme].items():
            net.all_airports[airport].ground_stop += constraint

    next_update_time = net.step()
    for i in range(9000):
        if i % 10 == 0:
            net.record_metrics()
        net.time += timedelta(minutes=1)
        if visualize:
            fig = net.make_figure()
            # fig.update_layout(
            #     {
            #         "paper_bgcolor": "rgba(0, 0, 0, 0)",
            #         "plot_bgcolor": "rgba(0, 0, 0, 0)",
            #     }
            # )
            fig.write_image(f'plots/{config}-{i}.png', format='png', width=1000, height=800, scale=2)
            # clear_output(wait=False)
            # fig.show()
        if net.time >= next_update_time:
            next_update_time = net.step()
            net.recover()
        if net.updates.empty():
            break

    net.write_report()
    with (net.report_dir / "meta.json").open(mode="w+") as meta_json:
        json.dump({
            "date": date,
            "crew_selection": crew_selection,
            "disruption": disruption,
            "ground_stop_scheme": ground_stop_scheme,
            "recovery": recovery,
            "crew_multiplier": crew_multiplier,
            "VERSION": VERSION,
            "generated_time": datetime.now().isoformat()
        }, meta_json)


def test_crew_window_empty():
    c = Crew(1, 'DAL')
    sch = Schedule('2022-12-24')
    flights = sch.schedule('N208WN')
    print(c.hours_remaining_after(flights.iloc[2], flights.iloc[0]['ScheduledDepTimeUTC'].to_pydatetime()))


if __name__ == "__main__":
    run_model()
    # test_crew_window_empty()
