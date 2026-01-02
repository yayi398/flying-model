#!/usr/bin/env python3
"""
Crew Scheduling Problem (CSP) Implementation using Genetic Algorithm
Based on the mathematical model described in README.md

This implementation schedules crew members to flights while satisfying:
- Minimum sit time between consecutive flights (30 minutes)
- Minimum rest time between duty days (10 hours = 600 minutes)
- Maximum flying time per duty (8 hours = 480 minutes)
- Maximum flying time in planning horizon (40 hours = 2400 minutes)
- Maximum elapsed/duty time per duty (12 hours = 720 minutes)
- Each flight needs a Captain and First Officer (2 crew members)

Extended features:
- Deadhead (空驶): Crew can travel as passenger to another city
  - Deadhead time does NOT count as flying time
  - Deadhead time counts as duty/work time
  - Any deadhead time can serve as rest time between consecutive operational flights
- Crew can end duty at non-home base (with hotel cost)
- 5-days-work / 2-days-rest: Crew must rest 2 days after every 5 consecutive working days
- No pairing generation limit
- Progress and timing output during solving
"""

import csv
import random
import copy
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field

# Constants based on typical aviation regulations
MIN_SIT_TIME = 30  # Minutes between consecutive flights on same duty day
MIN_REST_TIME = 600  # Minutes (10 hours) between duty days
MAX_FLYING_TIME_PER_DUTY = 480  # Minutes (8 hours) per duty day
MAX_FLYING_TIME_HORIZON = 2400  # Minutes (40 hours) in planning horizon
MAX_DUTY_TIME_PER_DUTY = 720  # Minutes (12 hours) per duty (including deadhead)
MAX_CONSECUTIVE_WORK_DAYS = 5  # Must rest after 5 consecutive working days
MIN_REST_DAYS_AFTER_WORK = 2  # Must rest 2 days after 5 consecutive work days

# GA Parameters (from README.md) - default values
DEFAULT_MAX_ITERATIONS = 150
DEFAULT_POPULATION_SIZE = 250
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.25

# Cost parameters
UNCOVERED_FLIGHT_COST = 10000  # High penalty for uncovered flights
DEADHEAD_COST = 500  # Cost for deadhead flights (per deadhead flight)
HOTEL_COST = 200  # Cost for crew staying overnight away from base (per night)
CREW_USAGE_COST = 100  # Base cost for using a crew member


def get_adaptive_ga_params(num_flights: int, num_crew: int) -> Tuple[int, int]:
    """
    Adjust GA parameters based on dataset size for better performance.
    Returns (max_iterations, population_size)
    NOTE: No longer limits pairings - unlimited pairing generation
    """
    if num_flights > 10000 or num_crew > 300:
        # Very large dataset
        return 20, 30
    elif num_flights > 5000 or num_crew > 200:
        # Large dataset
        return 30, 50
    elif num_flights > 1000 or num_crew > 50:
        # Medium dataset
        return 80, 100
    else:
        # Small dataset: use default parameters
        return DEFAULT_MAX_ITERATIONS, DEFAULT_POPULATION_SIZE


@dataclass
class CrewMember:
    """Represents a crew member"""
    emp_no: str
    is_captain: bool
    is_first_officer: bool
    can_deadhead: bool
    base: str
    duty_cost_per_hour: float
    pairing_cost_per_hour: float
    
    def __hash__(self):
        return hash(self.emp_no)


@dataclass
class Flight:
    """Represents a flight"""
    flight_num: str
    departure_date: datetime
    departure_time: datetime
    departure_station: str
    arrival_date: datetime
    arrival_time: datetime
    arrival_station: str
    comp: str
    flight_id: int = 0  # Unique identifier
    
    @property
    def duration_minutes(self) -> int:
        """Calculate flight duration in minutes"""
        delta = datetime.combine(self.arrival_date.date(), self.arrival_time.time()) - \
                datetime.combine(self.departure_date.date(), self.departure_time.time())
        return int(delta.total_seconds() / 60)
    
    def __hash__(self):
        return hash(self.flight_id)


@dataclass
class FlightLeg:
    """A single flight leg with its role (operational or deadhead)"""
    flight: Flight
    is_deadhead: bool = False  # True if crew is passenger (deadheading)
    
    @property
    def duration_minutes(self) -> int:
        return self.flight.duration_minutes


@dataclass
class Pairing:
    """A pairing is a sequence of flight legs that can be assigned to a crew member.
    
    Legs can be operational (crew flying) or deadhead (crew as passenger).
    Deadhead time does NOT count as flying time but DOES count as duty time.
    """
    legs: List[FlightLeg]
    start_city: str
    end_city: str
    
    @property
    def flights(self) -> List[Flight]:
        """Get all flights (for backward compatibility)"""
        return [leg.flight for leg in self.legs]
    
    @property
    def operational_flights(self) -> List[Flight]:
        """Get only operational flights (non-deadhead)"""
        return [leg.flight for leg in self.legs if not leg.is_deadhead]
    
    @property
    def deadhead_flights(self) -> List[Flight]:
        """Get only deadhead flights"""
        return [leg.flight for leg in self.legs if leg.is_deadhead]
    
    @property
    def total_flying_time(self) -> int:
        """Total flying time in minutes (EXCLUDES deadhead)"""
        return sum(leg.duration_minutes for leg in self.legs if not leg.is_deadhead)
    
    @property
    def total_deadhead_time(self) -> int:
        """Total deadhead time in minutes"""
        return sum(leg.duration_minutes for leg in self.legs if leg.is_deadhead)
    
    @property
    def total_duty_time(self) -> int:
        """Total duty time in minutes (includes deadhead)"""
        if not self.legs:
            return 0
        first_flight = self.legs[0].flight
        last_flight = self.legs[-1].flight
        start = datetime.combine(first_flight.departure_date.date(), first_flight.departure_time.time())
        end = datetime.combine(last_flight.arrival_date.date(), last_flight.arrival_time.time())
        return int((end - start).total_seconds() / 60)
    
    @property
    def elapsed_time(self) -> int:
        """Elapsed time from first departure to last arrival in minutes"""
        return self.total_duty_time
    
    @property
    def num_deadhead_legs(self) -> int:
        """Number of deadhead legs in this pairing"""
        return sum(1 for leg in self.legs if leg.is_deadhead)
    
    @property
    def requires_hotel(self) -> bool:
        """Check if crew ends at different city than they started"""
        return self.start_city != self.end_city


@dataclass
class CrewFlightAssignment:
    """Assignment of a crew member to a specific flight with role"""
    crew_emp_no: str
    flight: Flight
    role: str  # 'Captain', 'FirstOfficer', or 'Deadhead' (空驶)


@dataclass
class CrewAssignment:
    """Assignment of a crew member to a pairing with role"""
    crew: CrewMember
    pairing: Pairing
    role: str  # 'Captain' or 'FirstOfficer' (base role for operational flights)


@dataclass
class Schedule:
    """Complete schedule for all flights"""
    assignments: List[CrewAssignment] = field(default_factory=list)
    uncovered_flights: Set[int] = field(default_factory=set)
    
    def get_crew_assignments(self, crew: CrewMember) -> List[CrewAssignment]:
        """Get all assignments for a specific crew member"""
        return [a for a in self.assignments if a.crew == crew]


def parse_time(time_str: str) -> datetime:
    """Parse time string like '8:00' or '10:10' to datetime"""
    parts = time_str.split(':')
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return datetime(2021, 1, 1, hour, minute)


def parse_date(date_str: str) -> datetime:
    """Parse date string like '8/12/2021' to datetime"""
    parts = date_str.split('/')
    month = int(parts[0])
    day = int(parts[1])
    year = int(parts[2])
    return datetime(year, month, day)


def load_crew_data(filepath: str) -> List[CrewMember]:
    """Load crew members from CSV file"""
    crew_members = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column name variants between datasets A and B
            duty_cost = row.get('DutyCostPerHour', row.get('DutyCostPerHr', '0'))
            pairing_cost = row.get('PairingCostPerHour', row.get('ParingCostPerHour', 
                                   row.get('PairingCostPerHr', row.get('ParingCostPerHr', '0'))))
            
            crew = CrewMember(
                emp_no=row['EmpNo'],
                is_captain=row['Captain'] == 'Y',
                is_first_officer=row['FirstOfficer'] == 'Y',
                can_deadhead=row['Deadhead'] == 'Y',
                base=row['Base'],
                duty_cost_per_hour=float(duty_cost),
                pairing_cost_per_hour=float(pairing_cost)
            )
            crew_members.append(crew)
    return crew_members


def load_flight_data(filepath: str) -> List[Flight]:
    """Load flights from CSV file"""
    flights = []
    flight_id = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dept_date = parse_date(row['DptrDate'])
            dept_time = parse_time(row['DptrTime'])
            arrv_date = parse_date(row['ArrvDate'])
            arrv_time = parse_time(row['ArrvTime'])
            
            flight = Flight(
                flight_num=row['FltNum'],
                departure_date=dept_date,
                departure_time=datetime.combine(dept_date.date(), dept_time.time()),
                departure_station=row['DptrStn'],
                arrival_date=arrv_date,
                arrival_time=datetime.combine(arrv_date.date(), arrv_time.time()),
                arrival_station=row['ArrvStn'],
                comp=row['Comp'],
                flight_id=flight_id
            )
            flights.append(flight)
            flight_id += 1
    return flights


def can_connect_flights(flight1: Flight, flight2: Flight) -> bool:
    """Check if two flights can be connected in the same duty period.
    
    Conditions:
    - flight2 must depart from where flight1 arrives
    - Must have minimum sit time (30 min) between flights
    """
    # Flight 2 must depart from where flight 1 arrives
    if flight1.arrival_station != flight2.departure_station:
        return False
    
    # Calculate time between flights
    arrival = datetime.combine(flight1.arrival_date.date(), flight1.arrival_time.time())
    departure = datetime.combine(flight2.departure_date.date(), flight2.departure_time.time())
    
    sit_time = (departure - arrival).total_seconds() / 60
    
    # Must have at least minimum sit time
    return sit_time >= MIN_SIT_TIME


def can_connect_with_deadhead_rest(flight1: Flight, deadhead_flight: Flight, flight2: Flight) -> bool:
    """Check if deadhead flight between two operational flights provides sufficient rest.
    
    The rule: if crew flies flight1, then deadheads on deadhead_flight, then flies flight2,
    ANY deadhead time can count as rest time between the two operational flights.
    The total gap from flight1 arrival to flight2 departure (including deadhead) must satisfy
    the minimum sit time constraint.
    """
    # Deadhead must connect properly
    if flight1.arrival_station != deadhead_flight.departure_station:
        return False
    if deadhead_flight.arrival_station != flight2.departure_station:
        return False
    
    # Check timing
    f1_arrival = datetime.combine(flight1.arrival_date.date(), flight1.arrival_time.time())
    dh_departure = datetime.combine(deadhead_flight.departure_date.date(), deadhead_flight.departure_time.time())
    dh_arrival = datetime.combine(deadhead_flight.arrival_date.date(), deadhead_flight.arrival_time.time())
    f2_departure = datetime.combine(flight2.departure_date.date(), flight2.departure_time.time())
    
    # Sit time before deadhead
    sit_before_dh = (dh_departure - f1_arrival).total_seconds() / 60
    if sit_before_dh < MIN_SIT_TIME:
        return False
    
    # Sit time after deadhead to flight2
    sit_after_dh = (f2_departure - dh_arrival).total_seconds() / 60
    if sit_after_dh < MIN_SIT_TIME:
        return False
    
    # Any deadhead duration counts as rest - no minimum duration requirement
    # The deadhead time plus sit times contribute to the gap between operational flights
    return True


def generate_pairings(flights: List[Flight], base_city: str, all_cities: Set[str],
                      progress_interval: int = 1000) -> List[Pairing]:
    """
    Generate feasible pairings with support for:
    - Starting from base city
    - Ending at ANY city (with hotel cost if not base)
    - Deadhead legs (crew as passenger) to connect flights
    - No pairing limit
    
    Args:
        flights: List of all flights
        base_city: Home base city for crew
        all_cities: Set of all cities in the network
        progress_interval: Print progress every N pairings
    
    Returns:
        List of feasible pairings
    """
    pairings = []
    start_time = time.time()
    
    # Group flights by date for efficient lookup
    flights_by_date = {}
    for flight in flights:
        date_key = flight.departure_date.date()
        if date_key not in flights_by_date:
            flights_by_date[date_key] = []
        flights_by_date[date_key].append(flight)
    
    # Pre-index flights by departure station and date for faster lookups
    flights_by_station_date = {}
    for flight in flights:
        key = (flight.departure_station, flight.departure_date.date())
        if key not in flights_by_station_date:
            flights_by_station_date[key] = []
        flights_by_station_date[key].append(flight)
    
    # Pre-index flights by arrival station and date
    flights_by_arrival_date = {}
    for flight in flights:
        key = (flight.arrival_station, flight.arrival_date.date())
        if key not in flights_by_arrival_date:
            flights_by_arrival_date[key] = []
        flights_by_arrival_date[key].append(flight)
    
    def find_connecting_flights(prev_flight: Flight) -> List[Flight]:
        """Find flights that can connect after prev_flight"""
        key = (prev_flight.arrival_station, prev_flight.arrival_date.date())
        candidates = flights_by_station_date.get(key, [])
        return [f for f in candidates if can_connect_flights(prev_flight, f)]
    
    def find_deadhead_to_city(from_city: str, to_city: str, from_date, min_depart_time: datetime) -> List[Flight]:
        """Find flights that can be used as deadhead from one city to another"""
        key = (from_city, from_date)
        candidates = flights_by_station_date.get(key, [])
        result = []
        for f in candidates:
            if f.arrival_station == to_city:
                f_depart = datetime.combine(f.departure_date.date(), f.departure_time.time())
                if f_depart >= min_depart_time:
                    result.append(f)
        return result
    
    print(f"  Generating pairings from base {base_city}...")
    
    # === Generate single-flight pairings (for any flight departing from base) ===
    for date, day_flights in flights_by_date.items():
        outbound = [f for f in day_flights if f.departure_station == base_city]
        for flight in outbound:
            leg = FlightLeg(flight=flight, is_deadhead=False)
            pairing = Pairing(
                legs=[leg],
                start_city=base_city,
                end_city=flight.arrival_station
            )
            # Check constraints
            if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                pairings.append(pairing)
                if len(pairings) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
    
    # === Generate 2-flight pairings (base -> city -> anywhere) ===
    for date, day_flights in flights_by_date.items():
        outbound = [f for f in day_flights if f.departure_station == base_city]
        
        for f1 in outbound:
            connecting = find_connecting_flights(f1)
            
            for f2 in connecting:
                legs = [
                    FlightLeg(flight=f1, is_deadhead=False),
                    FlightLeg(flight=f2, is_deadhead=False)
                ]
                pairing = Pairing(
                    legs=legs,
                    start_city=base_city,
                    end_city=f2.arrival_station
                )
                if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                    pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                    pairings.append(pairing)
                    if len(pairings) % progress_interval == 0:
                        elapsed = time.time() - start_time
                        print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
    
    # === Generate pairings with deadhead to reach a flight ===
    # Pattern: deadhead from base to city X, then fly operational flight(s) from city X
    for date, day_flights in flights_by_date.items():
        # Find flights NOT departing from base
        non_base_flights = [f for f in day_flights if f.departure_station != base_city]
        
        for target_flight in non_base_flights:
            target_city = target_flight.departure_station
            target_depart = datetime.combine(target_flight.departure_date.date(), 
                                            target_flight.departure_time.time())
            
            # Find deadhead flights from base to target city
            # Need to arrive at target city with enough time before target flight
            min_dh_arrival = target_depart - timedelta(minutes=MIN_SIT_TIME)
            
            deadhead_candidates = find_deadhead_to_city(base_city, target_city, date, 
                                                        datetime(date.year, date.month, date.day, 0, 0))
            
            for dh_flight in deadhead_candidates[:10]:  # Limit to avoid explosion
                dh_arrival = datetime.combine(dh_flight.arrival_date.date(), dh_flight.arrival_time.time())
                if dh_arrival > min_dh_arrival:
                    continue  # Would not make it in time
                
                # Check sit time
                sit_time = (target_depart - dh_arrival).total_seconds() / 60
                if sit_time < MIN_SIT_TIME:
                    continue
                
                # Create pairing: deadhead + target flight
                legs = [
                    FlightLeg(flight=dh_flight, is_deadhead=True),
                    FlightLeg(flight=target_flight, is_deadhead=False)
                ]
                pairing = Pairing(
                    legs=legs,
                    start_city=base_city,
                    end_city=target_flight.arrival_station
                )
                if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                    pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                    pairings.append(pairing)
                    if len(pairings) % progress_interval == 0:
                        elapsed = time.time() - start_time
                        print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
                
                # Also try: deadhead + target + connecting flight
                connecting = find_connecting_flights(target_flight)
                for f2 in connecting[:5]:  # Limit
                    legs = [
                        FlightLeg(flight=dh_flight, is_deadhead=True),
                        FlightLeg(flight=target_flight, is_deadhead=False),
                        FlightLeg(flight=f2, is_deadhead=False)
                    ]
                    pairing = Pairing(
                        legs=legs,
                        start_city=base_city,
                        end_city=f2.arrival_station
                    )
                    if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                        pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                        pairings.append(pairing)
                        if len(pairings) % progress_interval == 0:
                            elapsed = time.time() - start_time
                            print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
    
    # === Generate 3-4 flight pairings (for smaller datasets) ===
    if len(flights) <= 2000:
        for date, day_flights in flights_by_date.items():
            outbound1 = [f for f in day_flights if f.departure_station == base_city]
            
            for f1 in outbound1:
                connecting_f1 = find_connecting_flights(f1)
                
                for f2 in connecting_f1:
                    connecting_f2 = find_connecting_flights(f2)
                    
                    for f3 in connecting_f2:
                        # 3-flight pairing
                        legs = [
                            FlightLeg(flight=f1, is_deadhead=False),
                            FlightLeg(flight=f2, is_deadhead=False),
                            FlightLeg(flight=f3, is_deadhead=False)
                        ]
                        pairing = Pairing(
                            legs=legs,
                            start_city=base_city,
                            end_city=f3.arrival_station
                        )
                        if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                            pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                            pairings.append(pairing)
                            if len(pairings) % progress_interval == 0:
                                elapsed = time.time() - start_time
                                print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
                        
                        # 4-flight pairings
                        if len(flights) <= 500:
                            connecting_f3 = find_connecting_flights(f3)
                            for f4 in connecting_f3:
                                legs = [
                                    FlightLeg(flight=f1, is_deadhead=False),
                                    FlightLeg(flight=f2, is_deadhead=False),
                                    FlightLeg(flight=f3, is_deadhead=False),
                                    FlightLeg(flight=f4, is_deadhead=False)
                                ]
                                pairing = Pairing(
                                    legs=legs,
                                    start_city=base_city,
                                    end_city=f4.arrival_station
                                )
                                if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                                    pairing.total_duty_time <= MAX_DUTY_TIME_PER_DUTY):
                                    pairings.append(pairing)
                                    if len(pairings) % progress_interval == 0:
                                        elapsed = time.time() - start_time
                                        print(f"    Generated {len(pairings)} pairings... ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    print(f"  Base {base_city}: Generated {len(pairings)} pairings in {elapsed:.1f}s")
    return pairings


def get_flight_pairings_map(pairings: List[Pairing]) -> Dict[int, List[int]]:
    """Create a map from flight_id to list of pairing indices that contain it (operational flights only)"""
    flight_to_pairings = {}
    for i, pairing in enumerate(pairings):
        # Only map operational flights (non-deadhead)
        for flight in pairing.operational_flights:
            if flight.flight_id not in flight_to_pairings:
                flight_to_pairings[flight.flight_id] = []
            flight_to_pairings[flight.flight_id].append(i)
    return flight_to_pairings


def check_consecutive_work_days_constraint(work_days: Dict, new_date) -> bool:
    """
    Check if assigning a new work day would violate the 5-consecutive-days / 2-days-rest constraint.
    
    Rules:
    - Crew can work at most 5 consecutive days
    - After 5 consecutive days, crew must rest for 2 days
    
    Args:
        work_days: Dictionary mapping dates to True for days worked
        new_date: The proposed new work date
    
    Returns:
        True if the new assignment is allowed, False if it would violate the constraint
    """
    if not work_days:
        return True
    
    # Convert work_days keys to a sorted list of dates
    worked_dates = sorted(work_days.keys())
    
    # Create a set for O(1) lookup
    worked_set = set(worked_dates)
    
    # Add the new date temporarily
    test_dates = worked_set | {new_date}
    
    # Check for any sequence of more than MAX_CONSECUTIVE_WORK_DAYS consecutive days
    all_dates = sorted(test_dates)
    
    # Track consecutive work days
    consecutive = 1
    for i in range(1, len(all_dates)):
        # Check if this date is consecutive to the previous one
        if (all_dates[i] - all_dates[i-1]).days == 1:
            consecutive += 1
            if consecutive > MAX_CONSECUTIVE_WORK_DAYS:
                return False  # Would exceed 5 consecutive days
        else:
            # Check if there's a gap and if rest was sufficient after a 5-day streak
            gap_days = (all_dates[i] - all_dates[i-1]).days - 1
            if consecutive == MAX_CONSECUTIVE_WORK_DAYS and gap_days < MIN_REST_DAYS_AFTER_WORK:
                # Worked 5 days but didn't rest enough before working again
                return False
            consecutive = 1
    
    return True


def calculate_fitness(chromosome: List[float], pairings: List[Pairing], 
                     flights: List[Flight], flight_to_pairings: Dict[int, List[int]],
                     captains: List[CrewMember], first_officers: List[CrewMember]) -> Tuple[float, Schedule]:
    """
    Calculate fitness of a chromosome and return the corresponding schedule.
    Lower fitness is better (cost minimization).
    
    Optimized for large datasets with efficient greedy assignment.
    """
    schedule = Schedule()
    covered_flights = set()
    crew_total_flying = {c.emp_no: 0 for c in captains + first_officers}
    crew_assignments_per_day = {c.emp_no: {} for c in captains + first_officers}
    
    # For large datasets, use a sampling approach to speed up
    num_flights = len(flights)
    use_sampling = num_flights > 5000
    
    # Sort flights by number of available pairings (ascending) - prioritize harder flights
    if use_sampling:
        # For large datasets, sample a subset of flights for ordering
        sample_size = min(2000, num_flights)
        import random as random_module
        sample_indices = random_module.sample(range(num_flights), sample_size)
        flight_order = sorted(sample_indices, 
                             key=lambda fid: len(flight_to_pairings.get(fid, [])))
        # Add remaining flights
        remaining = [i for i in range(num_flights) if i not in set(sample_indices)]
        flight_order.extend(remaining)
    else:
        flight_order = sorted(range(num_flights), 
                             key=lambda fid: len(flight_to_pairings.get(fid, [])))
    
    # Assign pairings based on chromosome
    captain_idx = 0
    fo_idx = 0
    
    # Limit iterations for large datasets
    max_iterations = min(len(flight_order), 50000)
    
    for i, flight_idx in enumerate(flight_order):
        if i >= max_iterations:
            break
            
        if flight_idx in covered_flights:
            continue
            
        available_pairings = flight_to_pairings.get(flight_idx, [])
        if not available_pairings:
            schedule.uncovered_flights.add(flight_idx)
            continue
        
        # Use chromosome value to select pairing
        gene_idx = flight_idx % len(chromosome)
        pairing_local_idx = min(int(chromosome[gene_idx] * len(available_pairings)), len(available_pairings) - 1)
        selected_pairing_idx = available_pairings[pairing_local_idx]
        selected_pairing = pairings[selected_pairing_idx]
        
        # Check if all operational flights in pairing are still available
        pairing_op_flights_ids = [f.flight_id for f in selected_pairing.operational_flights]
        if any(fid in covered_flights for fid in pairing_op_flights_ids):
            # Try to find another pairing (limit search)
            found = False
            search_limit = min(len(available_pairings), 10)  # Limit search
            for alt_idx in available_pairings[:search_limit]:
                alt_pairing = pairings[alt_idx]
                alt_flight_ids = [f.flight_id for f in alt_pairing.operational_flights]
                if not any(fid in covered_flights for fid in alt_flight_ids):
                    selected_pairing = alt_pairing
                    found = True
                    break
            if not found:
                continue
        
        # Assign captain (limit search)
        assigned_captain = None
        search_limit = min(len(captains), 50)
        for _ in range(search_limit):
            captain = captains[captain_idx % len(captains)]
            captain_idx += 1
            
            # Check if captain is available and constraints are satisfied
            pairing_date = selected_pairing.legs[0].flight.departure_date.date()
            if pairing_date in crew_assignments_per_day[captain.emp_no]:
                continue
            
            if crew_total_flying[captain.emp_no] + selected_pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
                continue
            
            # Check 5-consecutive-days constraint
            if not check_consecutive_work_days_constraint(crew_assignments_per_day[captain.emp_no], pairing_date):
                continue
            
            assigned_captain = captain
            break
        
        # Assign first officer (limit search)
        assigned_fo = None
        search_limit = min(len(first_officers), 50)
        for _ in range(search_limit):
            fo = first_officers[fo_idx % len(first_officers)]
            fo_idx += 1
            
            # Skip if same as assigned captain
            if assigned_captain and fo.emp_no == assigned_captain.emp_no:
                continue
            
            pairing_date = selected_pairing.legs[0].flight.departure_date.date()
            if pairing_date in crew_assignments_per_day[fo.emp_no]:
                continue
            
            if crew_total_flying[fo.emp_no] + selected_pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
                continue
            
            # Check 5-consecutive-days constraint
            if not check_consecutive_work_days_constraint(crew_assignments_per_day[fo.emp_no], pairing_date):
                continue
            
            assigned_fo = fo
            break
        
        if assigned_captain and assigned_fo:
            # Create assignments
            captain_assignment = CrewAssignment(
                crew=assigned_captain,
                pairing=selected_pairing,
                role='Captain'
            )
            fo_assignment = CrewAssignment(
                crew=assigned_fo,
                pairing=selected_pairing,
                role='FirstOfficer'
            )
            
            schedule.assignments.append(captain_assignment)
            schedule.assignments.append(fo_assignment)
            
            # Update tracking - only cover operational flights
            for flight in selected_pairing.operational_flights:
                covered_flights.add(flight.flight_id)
            
            pairing_date = selected_pairing.legs[0].flight.departure_date.date()
            crew_total_flying[assigned_captain.emp_no] += selected_pairing.total_flying_time
            crew_total_flying[assigned_fo.emp_no] += selected_pairing.total_flying_time
            crew_assignments_per_day[assigned_captain.emp_no][pairing_date] = True
            crew_assignments_per_day[assigned_fo.emp_no][pairing_date] = True
        else:
            schedule.uncovered_flights.add(flight_idx)
    
    # Mark remaining uncovered flights
    for flight in flights:
        if flight.flight_id not in covered_flights:
            schedule.uncovered_flights.add(flight.flight_id)
    
    # Calculate total cost
    cost = len(schedule.uncovered_flights) * UNCOVERED_FLIGHT_COST
    
    # Add crew usage cost
    used_crew = set(a.crew.emp_no for a in schedule.assignments)
    cost += len(used_crew) * CREW_USAGE_COST
    
    # Add flying time cost, deadhead cost, and hotel cost
    for assignment in schedule.assignments:
        hours = assignment.pairing.total_flying_time / 60
        cost += hours * assignment.crew.duty_cost_per_hour
        # Add deadhead cost
        cost += assignment.pairing.num_deadhead_legs * DEADHEAD_COST
        # Add hotel cost if ending away from home base
        if assignment.pairing.requires_hotel:
            cost += HOTEL_COST
    
    return cost, schedule


def greedy_solve(pairings: List[Pairing], flights: List[Flight],
                 captains: List[CrewMember], first_officers: List[CrewMember]) -> Tuple[Schedule, float]:
    """
    Greedy solver for large datasets - faster than GA but may not find optimal solution.
    Sorts pairings by number of operational flights covered and greedily assigns crew.
    """
    start_time = time.time()
    print("Using greedy solver...")
    
    schedule = Schedule()
    covered_flights = set()
    crew_total_flying = {c.emp_no: 0 for c in captains + first_officers}
    crew_assignments_per_day = {c.emp_no: {} for c in captains + first_officers}
    
    # Sort pairings by number of operational flights (descending) to maximize coverage
    # Prefer pairings without deadhead (cheaper), then by more operational flights
    sorted_pairings = sorted(pairings, 
                            key=lambda p: (len(p.operational_flights), -p.num_deadhead_legs), 
                            reverse=True)
    
    captain_idx = 0
    fo_idx = 0
    
    total_pairings = len(sorted_pairings)
    for i, pairing in enumerate(sorted_pairings):
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processing pairing {i+1}/{total_pairings}, covered: {len(covered_flights)}/{len(flights)}... ({elapsed:.1f}s)", end='\r')
        
        # Check if any operational flight in pairing is already covered
        pairing_op_flights_ids = [f.flight_id for f in pairing.operational_flights]
        if any(fid in covered_flights for fid in pairing_op_flights_ids):
            continue
        
        # Try to assign captain
        assigned_captain = None
        pairing_date = pairing.legs[0].flight.departure_date.date()
        
        for _ in range(min(len(captains), 100)):
            captain = captains[captain_idx % len(captains)]
            captain_idx += 1
            
            if pairing_date in crew_assignments_per_day.get(captain.emp_no, {}):
                continue
            if crew_total_flying.get(captain.emp_no, 0) + pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
                continue
            # Check 5-consecutive-days constraint
            if not check_consecutive_work_days_constraint(crew_assignments_per_day.get(captain.emp_no, {}), pairing_date):
                continue
            
            assigned_captain = captain
            break
        
        if not assigned_captain:
            continue
        
        # Try to assign first officer
        assigned_fo = None
        for _ in range(min(len(first_officers), 100)):
            fo = first_officers[fo_idx % len(first_officers)]
            fo_idx += 1
            
            if fo.emp_no == assigned_captain.emp_no:
                continue
            if pairing_date in crew_assignments_per_day.get(fo.emp_no, {}):
                continue
            if crew_total_flying.get(fo.emp_no, 0) + pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
                continue
            # Check 5-consecutive-days constraint
            if not check_consecutive_work_days_constraint(crew_assignments_per_day.get(fo.emp_no, {}), pairing_date):
                continue
            
            assigned_fo = fo
            break
        
        if not assigned_fo:
            continue
        
        # Create assignments
        schedule.assignments.append(CrewAssignment(
            crew=assigned_captain,
            pairing=pairing,
            role='Captain'
        ))
        schedule.assignments.append(CrewAssignment(
            crew=assigned_fo,
            pairing=pairing,
            role='FirstOfficer'
        ))
        
        # Update tracking - only cover operational flights
        for flight in pairing.operational_flights:
            covered_flights.add(flight.flight_id)
        
        crew_total_flying[assigned_captain.emp_no] += pairing.total_flying_time
        crew_total_flying[assigned_fo.emp_no] += pairing.total_flying_time
        crew_assignments_per_day[assigned_captain.emp_no][pairing_date] = True
        crew_assignments_per_day[assigned_fo.emp_no][pairing_date] = True
    
    elapsed = time.time() - start_time
    print(f"\n  Greedy solve completed in {elapsed:.1f}s")
    
    # Mark uncovered flights
    for flight in flights:
        if flight.flight_id not in covered_flights:
            schedule.uncovered_flights.add(flight.flight_id)
    
    # Calculate cost
    cost = len(schedule.uncovered_flights) * UNCOVERED_FLIGHT_COST
    used_crew = set(a.crew.emp_no for a in schedule.assignments)
    cost += len(used_crew) * CREW_USAGE_COST
    
    for assignment in schedule.assignments:
        hours = assignment.pairing.total_flying_time / 60
        cost += hours * assignment.crew.duty_cost_per_hour
        # Add deadhead cost
        cost += assignment.pairing.num_deadhead_legs * DEADHEAD_COST
        # Add hotel cost if ending away from home base
        if assignment.pairing.requires_hotel:
            cost += HOTEL_COST
    
    return schedule, cost


class GeneticAlgorithm:
    """Genetic Algorithm for Crew Scheduling Problem"""
    
    def __init__(self, pairings: List[Pairing], flights: List[Flight],
                 captains: List[CrewMember], first_officers: List[CrewMember],
                 max_iterations: int = DEFAULT_MAX_ITERATIONS, 
                 population_size: int = DEFAULT_POPULATION_SIZE):
        self.pairings = pairings
        self.flights = flights
        self.captains = captains
        self.first_officers = first_officers
        self.flight_to_pairings = get_flight_pairings_map(pairings)
        self.chromosome_length = len(flights)
        self.max_iterations = max_iterations
        self.population_size = population_size
        
    def initialize_population(self) -> List[List[float]]:
        """Initialize population with random chromosomes"""
        population = []
        for _ in range(self.population_size):
            chromosome = [random.random() for _ in range(self.chromosome_length)]
            population.append(chromosome)
        return population
    
    def evaluate_population(self, population: List[List[float]]) -> List[Tuple[float, Schedule, List[float]]]:
        """Evaluate fitness of all chromosomes"""
        results = []
        total = len(population)
        large_dataset = len(self.flights) > 5000
        for i, chromosome in enumerate(population):
            if large_dataset and i % 10 == 0:
                print(f"  Evaluating chromosome {i+1}/{total}...", end='\r')
            fitness, schedule = calculate_fitness(
                chromosome, self.pairings, self.flights,
                self.flight_to_pairings, self.captains, self.first_officers
            )
            results.append((fitness, schedule, chromosome))
        if large_dataset:
            print()  # New line after progress
        return sorted(results, key=lambda x: x[0])
    
    def roulette_wheel_selection(self, evaluated_pop: List[Tuple[float, Schedule, List[float]]]) -> List[float]:
        """Select parent using ranking-based selection (more robust than fitness proportionate)"""
        # Use ranking-based selection to avoid numerical issues with large fitness values
        n = len(evaluated_pop)
        # Rank-based weights: higher rank (lower fitness) gets more weight
        rank_weights = [n - i for i in range(n)]
        total = sum(rank_weights)
        
        r = random.random() * total
        cumulative = 0
        for i, weight in enumerate(rank_weights):
            cumulative += weight
            if cumulative >= r:
                return evaluated_pop[i][2]
        return evaluated_pop[-1][2]
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Multi-point crossover"""
        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()
        
        # Handle short chromosomes
        if len(parent1) <= 2:
            return parent1.copy(), parent2.copy()
        
        # Select multiple crossover points
        num_points = random.randint(1, 3)
        max_points = min(num_points, len(parent1) - 1)
        if max_points < 1:
            return parent1.copy(), parent2.copy()
        points = sorted(random.sample(range(1, len(parent1)), max_points))
        
        child1 = []
        child2 = []
        use_parent1 = True
        
        prev_point = 0
        for point in points + [len(parent1)]:
            if use_parent1:
                child1.extend(parent1[prev_point:point])
                child2.extend(parent2[prev_point:point])
            else:
                child1.extend(parent2[prev_point:point])
                child2.extend(parent1[prev_point:point])
            use_parent1 = not use_parent1
            prev_point = point
        
        return child1, child2
    
    def mutate(self, chromosome: List[float]) -> List[float]:
        """Mutation: replace some genes with new random values"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[i] = random.random()
        return mutated
    
    def run(self) -> Tuple[Schedule, float]:
        """Run the genetic algorithm"""
        population = self.initialize_population()
        best_fitness = float('inf')
        best_schedule = None
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            evaluated_pop = self.evaluate_population(population)
            
            # Track best solution
            if evaluated_pop[0][0] < best_fitness:
                best_fitness = evaluated_pop[0][0]
                best_schedule = evaluated_pop[0][1]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early termination if no improvement
            if no_improvement_count > 20:
                print(f"Early termination at iteration {iteration}")
                break
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {best_fitness:.2f}, "
                      f"Uncovered flights = {len(best_schedule.uncovered_flights)}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top 10% of population
            elite_size = max(1, self.population_size // 10)
            for i in range(elite_size):
                new_population.append(evaluated_pop[i][2])
            
            # Fill rest with offspring
            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(evaluated_pop)
                parent2 = self.roulette_wheel_selection(evaluated_pop)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        return best_schedule, best_fitness


def output_schedule_csv(schedule: Schedule, flights: List[Flight], output_path: str):
    """Output the schedule to a CSV file with crew roles (Captain/FirstOfficer/Deadhead)"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FlightNum', 'Date', 'DepartureTime', 'DepartureStation', 
                        'ArrivalTime', 'ArrivalStation', 'Captain', 'CaptainRole',
                        'FirstOfficer', 'FirstOfficerRole', 'DeadheadCrew'])
        
        # Create flight to crew mapping with roles
        # flight_id -> {'Captain': emp_no, 'CaptainRole': role, 'FirstOfficer': emp_no, 'FirstOfficerRole': role, 'Deadhead': [emp_nos]}
        flight_assignments = {}
        for assignment in schedule.assignments:
            for leg in assignment.pairing.legs:
                flight = leg.flight
                if flight.flight_id not in flight_assignments:
                    flight_assignments[flight.flight_id] = {
                        'Captain': '', 'CaptainRole': '',
                        'FirstOfficer': '', 'FirstOfficerRole': '',
                        'Deadhead': []
                    }
                
                if leg.is_deadhead:
                    # This crew is deadheading on this flight
                    flight_assignments[flight.flight_id]['Deadhead'].append(assignment.crew.emp_no)
                else:
                    # This is an operational flight for this crew
                    if assignment.role == 'Captain':
                        flight_assignments[flight.flight_id]['Captain'] = assignment.crew.emp_no
                        flight_assignments[flight.flight_id]['CaptainRole'] = 'Captain'
                    else:
                        flight_assignments[flight.flight_id]['FirstOfficer'] = assignment.crew.emp_no
                        flight_assignments[flight.flight_id]['FirstOfficerRole'] = 'FirstOfficer'
        
        # Output each flight
        for flight in sorted(flights, key=lambda f: (f.departure_date, f.departure_time)):
            fdata = flight_assignments.get(flight.flight_id, {})
            captain = fdata.get('Captain', 'UNCOVERED')
            captain_role = fdata.get('CaptainRole', '')
            fo = fdata.get('FirstOfficer', 'UNCOVERED')
            fo_role = fdata.get('FirstOfficerRole', '')
            deadhead = ','.join(fdata.get('Deadhead', []))
            
            writer.writerow([
                flight.flight_num,
                flight.departure_date.strftime('%Y-%m-%d'),
                flight.departure_time.strftime('%H:%M'),
                flight.departure_station,
                flight.arrival_time.strftime('%H:%M'),
                flight.arrival_station,
                captain if captain else 'UNCOVERED',
                captain_role,
                fo if fo else 'UNCOVERED',
                fo_role,
                deadhead
            ])
    
    print(f"Schedule saved to {output_path}")


def output_crew_schedule_csv(schedule: Schedule, crew_members: List[CrewMember], output_path: str):
    """Output crew-centric schedule to CSV file with detailed role information"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['CrewID', 'BaseRole', 'Date', 'Flights', 'FlightRoles',
                        'TotalFlyingTime(min)', 'TotalDutyTime(min)', 'DeadheadCount',
                        'StartCity', 'EndCity', 'HotelRequired'])
        
        # Group assignments by crew
        crew_pairings = {}
        for assignment in schedule.assignments:
            if assignment.crew.emp_no not in crew_pairings:
                crew_pairings[assignment.crew.emp_no] = []
            crew_pairings[assignment.crew.emp_no].append(assignment)
        
        # Output for each crew
        for crew in crew_members:
            if crew.emp_no in crew_pairings:
                for assignment in sorted(crew_pairings[crew.emp_no], 
                                        key=lambda a: a.pairing.legs[0].flight.departure_date):
                    # Build flight list with roles
                    flight_nums = []
                    flight_roles = []
                    for leg in assignment.pairing.legs:
                        flight_nums.append(leg.flight.flight_num)
                        if leg.is_deadhead:
                            flight_roles.append('Deadhead')
                        else:
                            flight_roles.append(assignment.role)
                    
                    date = assignment.pairing.legs[0].flight.departure_date.strftime('%Y-%m-%d')
                    
                    writer.writerow([
                        crew.emp_no,
                        assignment.role,
                        date,
                        ' -> '.join(flight_nums),
                        ' -> '.join(flight_roles),
                        assignment.pairing.total_flying_time,
                        assignment.pairing.total_duty_time,
                        assignment.pairing.num_deadhead_legs,
                        assignment.pairing.start_city,
                        assignment.pairing.end_city,
                        'Yes' if assignment.pairing.requires_hotel else 'No'
                    ])
    
    print(f"Crew schedule saved to {output_path}")


def print_schedule_summary(schedule: Schedule, flights: List[Flight], crew_members: List[CrewMember]):
    """Print a summary of the schedule"""
    print("\n" + "="*60)
    print("SCHEDULE SUMMARY")
    print("="*60)
    
    total_flights = len(flights)
    covered_flights = total_flights - len(schedule.uncovered_flights)
    
    print(f"Total flights: {total_flights}")
    print(f"Covered flights: {covered_flights}")
    print(f"Uncovered flights: {len(schedule.uncovered_flights)}")
    print(f"Coverage rate: {covered_flights/total_flights*100:.1f}%")
    
    # Count deadhead and hotel usage
    total_deadhead = 0
    total_hotel = 0
    for assignment in schedule.assignments:
        total_deadhead += assignment.pairing.num_deadhead_legs
        if assignment.pairing.requires_hotel:
            total_hotel += 1
    
    print(f"\nDeadhead flights used: {total_deadhead}")
    print(f"Hotel stays required: {total_hotel}")
    
    # Crew statistics
    crew_flying_time = {}
    crew_duty_time = {}
    crew_duty_days = {}
    for assignment in schedule.assignments:
        emp = assignment.crew.emp_no
        if emp not in crew_flying_time:
            crew_flying_time[emp] = 0
            crew_duty_time[emp] = 0
            crew_duty_days[emp] = set()
        crew_flying_time[emp] += assignment.pairing.total_flying_time
        crew_duty_time[emp] += assignment.pairing.total_duty_time
        crew_duty_days[emp].add(assignment.pairing.legs[0].flight.departure_date.date())
    
    print(f"\nCrew members used: {len(crew_flying_time)}/{len(crew_members)}")
    
    # Only print detailed utilization for small datasets
    if len(crew_flying_time) <= 30:
        print("\nCrew utilization (flying/duty hours):")
        for emp in sorted(crew_flying_time.keys()):
            flying_hours = crew_flying_time[emp] / 60
            duty_hours = crew_duty_time[emp] / 60
            days = len(crew_duty_days[emp])
            print(f"  {emp}: {flying_hours:.1f}/{duty_hours:.1f} hours over {days} duty days")
    else:
        # For large datasets, show summary statistics
        flying_times = list(crew_flying_time.values())
        avg_flying = sum(flying_times) / len(flying_times) / 60
        max_flying = max(flying_times) / 60
        min_flying = min(flying_times) / 60
        print(f"\nFlying time stats: avg={avg_flying:.1f}h, min={min_flying:.1f}h, max={max_flying:.1f}h")
    
    if schedule.uncovered_flights and len(schedule.uncovered_flights) <= 20:
        print(f"\nUncovered flight IDs: {sorted(schedule.uncovered_flights)}")


def main():
    """Main function to run the crew scheduling optimization"""
    import os
    import argparse
    
    total_start_time = time.time()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Crew Scheduling Optimization using Genetic Algorithm')
    parser.add_argument('--crew-file', default=os.path.join(script_dir, "机组排班Data A-Crew.csv"),
                       help='Path to crew data CSV file')
    parser.add_argument('--flight-file', default=os.path.join(script_dir, "机组排班Data A-Flight.csv"),
                       help='Path to flight data CSV file')
    parser.add_argument('--output-file', default=os.path.join(script_dir, "schedule_output.csv"),
                       help='Path to output schedule CSV file')
    parser.add_argument('--crew-output-file', default=os.path.join(script_dir, "crew_schedule_output.csv"),
                       help='Path to crew schedule output CSV file')
    
    args = parser.parse_args()
    
    # File paths
    crew_file = args.crew_file
    flight_file = args.flight_file
    output_file = args.output_file
    crew_output_file = args.crew_output_file
    
    print("="*60)
    print("CREW SCHEDULING OPTIMIZATION")
    print("="*60)
    
    print("\n[1/4] Loading data...")
    crew_members = load_crew_data(crew_file)
    flights = load_flight_data(flight_file)
    
    print(f"  Loaded {len(crew_members)} crew members")
    print(f"  Loaded {len(flights)} flights")
    
    # Separate captains and first officers
    captains = [c for c in crew_members if c.is_captain]
    first_officers = [c for c in crew_members if c.is_first_officer]
    
    print(f"  Captains: {len(captains)}")
    print(f"  First Officers: {len(first_officers)}")
    
    # Get all unique cities
    all_cities = set()
    for f in flights:
        all_cities.add(f.departure_station)
        all_cities.add(f.arrival_station)
    print(f"  Total cities: {len(all_cities)}")
    
    # Get all unique base cities from crew members
    base_cities = list(set(c.base for c in crew_members))
    print(f"  Base cities: {base_cities}")
    
    # Get adaptive GA parameters based on dataset size
    max_iterations, population_size = get_adaptive_ga_params(len(flights), len(crew_members))
    
    # Generate pairings for all base cities
    print("\n[2/4] Generating feasible pairings (no limit)...")
    pairing_start = time.time()
    all_pairings = []
    for base_city in base_cities:
        pairings = generate_pairings(flights, base_city, all_cities)
        all_pairings.extend(pairings)
    
    pairing_elapsed = time.time() - pairing_start
    print(f"\nTotal pairings generated: {len(all_pairings)} in {pairing_elapsed:.1f}s")
    
    if not all_pairings:
        print("ERROR: No feasible pairings found. Check flight data and constraints.")
        return
    
    # Count pairings with deadhead and hotel
    deadhead_pairings = sum(1 for p in all_pairings if p.num_deadhead_legs > 0)
    hotel_pairings = sum(1 for p in all_pairings if p.requires_hotel)
    print(f"  Pairings with deadhead: {deadhead_pairings}")
    print(f"  Pairings requiring hotel: {hotel_pairings}")
    
    # Use greedy solver for very large datasets, GA for smaller ones
    use_greedy = len(flights) > 5000 or len(crew_members) > 200
    
    print("\n[3/4] Running optimization...")
    opt_start = time.time()
    
    if use_greedy:
        best_schedule, best_fitness = greedy_solve(all_pairings, flights, captains, first_officers)
    else:
        print(f"  Using Genetic Algorithm")
        print(f"  Parameters: iterations={max_iterations}, population={population_size}, "
              f"crossover={CROSSOVER_RATE}, mutation={MUTATION_RATE}")
        
        ga = GeneticAlgorithm(all_pairings, flights, captains, first_officers, 
                              max_iterations, population_size)
        best_schedule, best_fitness = ga.run()
    
    opt_elapsed = time.time() - opt_start
    print(f"\n  Optimization completed in {opt_elapsed:.1f}s")
    print(f"  Best fitness (cost): {best_fitness:.2f}")
    
    # Print summary
    print_schedule_summary(best_schedule, flights, crew_members)
    
    # Output schedules
    print("\n[4/4] Saving output files...")
    output_schedule_csv(best_schedule, flights, output_file)
    output_crew_schedule_csv(best_schedule, crew_members, crew_output_file)
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_elapsed:.1f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
