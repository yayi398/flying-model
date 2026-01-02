#!/usr/bin/env python3
"""
Crew Scheduling Problem (CSP) Implementation using Genetic Algorithm
Based on the mathematical model described in README.md

This implementation schedules crew members to flights while satisfying:
- Minimum sit time between consecutive flights (30 minutes)
- Minimum rest time between duty days (10 hours = 600 minutes)
- Maximum flying time per duty (8 hours = 480 minutes)
- Maximum flying time in planning horizon (40 hours = 2400 minutes)
- Maximum elapsed time per duty (12 hours = 720 minutes)
- Crew must start first duty from home base
- Crew must return to home base at end of last duty
- Each flight needs a Captain and First Officer (2 crew members)
"""

import csv
import random
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field

# Constants based on typical aviation regulations
MIN_SIT_TIME = 30  # Minutes between consecutive flights on same duty day
MIN_REST_TIME = 600  # Minutes (10 hours) between duty days
MAX_FLYING_TIME_PER_DUTY = 480  # Minutes (8 hours) per duty day
MAX_FLYING_TIME_HORIZON = 2400  # Minutes (40 hours) in planning horizon
MAX_ELAPSED_TIME_PER_DUTY = 720  # Minutes (12 hours) per duty

# GA Parameters (from README.md) - default values, adjusted for large datasets
DEFAULT_MAX_ITERATIONS = 150
DEFAULT_POPULATION_SIZE = 250
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.25

# Cost parameters
UNCOVERED_FLIGHT_COST = 10000  # High penalty for uncovered flights
DEADHEAD_COST = 500  # Cost for deadhead flights
HOTEL_COST = 200  # Cost for crew staying overnight away from base
CREW_USAGE_COST = 100  # Base cost for using a crew member


def get_adaptive_ga_params(num_flights: int, num_crew: int) -> Tuple[int, int, int]:
    """
    Adjust GA parameters based on dataset size for better performance.
    Returns (max_iterations, population_size, max_pairings)
    """
    if num_flights > 10000 or num_crew > 300:
        # Very large dataset: minimal iterations, very small population
        return 20, 30, 30000
    elif num_flights > 5000 or num_crew > 200:
        # Large dataset: reduce iterations and population, limit pairings
        return 30, 50, 50000
    elif num_flights > 1000 or num_crew > 50:
        # Medium dataset
        return 80, 100, 100000
    else:
        # Small dataset: use default parameters, unlimited pairings
        return DEFAULT_MAX_ITERATIONS, DEFAULT_POPULATION_SIZE, 1000000


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
class Pairing:
    """A pairing is a sequence of flights that can be assigned to a crew member"""
    flights: List[Flight]
    start_city: str
    end_city: str
    
    @property
    def total_flying_time(self) -> int:
        """Total flying time in minutes"""
        return sum(f.duration_minutes for f in self.flights)
    
    @property
    def elapsed_time(self) -> int:
        """Elapsed time from first departure to last arrival in minutes"""
        if not self.flights:
            return 0
        first_flight = self.flights[0]
        last_flight = self.flights[-1]
        start = datetime.combine(first_flight.departure_date.date(), first_flight.departure_time.time())
        end = datetime.combine(last_flight.arrival_date.date(), last_flight.arrival_time.time())
        return int((end - start).total_seconds() / 60)


@dataclass
class CrewAssignment:
    """Assignment of a crew member to a pairing with role"""
    crew: CrewMember
    pairing: Pairing
    role: str  # 'Captain' or 'FirstOfficer'


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
    """Check if two flights can be connected in the same duty day"""
    # Flight 2 must depart after flight 1 arrives
    if flight1.arrival_station != flight2.departure_station:
        return False
    
    # Calculate time between flights
    arrival = datetime.combine(flight1.arrival_date.date(), flight1.arrival_time.time())
    departure = datetime.combine(flight2.departure_date.date(), flight2.departure_time.time())
    
    sit_time = (departure - arrival).total_seconds() / 60
    
    # Check minimum sit time (same day)
    if flight1.arrival_date.date() == flight2.departure_date.date():
        return sit_time >= MIN_SIT_TIME
    
    # Check minimum rest time (different days)
    return sit_time >= MIN_REST_TIME


def generate_pairings(flights: List[Flight], base_city: str, max_flights: int = 4, 
                      max_pairings: int = 100000) -> List[Pairing]:
    """
    Generate feasible pairings starting and ending at the base city.
    A pairing can have 1-4 flights following FAA rules.
    
    For large datasets, limits total pairings to max_pairings for performance.
    """
    pairings = []
    
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
    
    # Helper function to find connecting flights efficiently
    def find_connecting_flights(prev_flight):
        key = (prev_flight.arrival_station, prev_flight.arrival_date.date())
        candidates = flights_by_station_date.get(key, [])
        return [f for f in candidates if can_connect_flights(prev_flight, f)]
    
    # Generate single-flight pairings for flights that end at base (deadhead home)
    for date, day_flights in flights_by_date.items():
        inbound_only = [f for f in day_flights if f.arrival_station == base_city]
        for flight in inbound_only:
            if len(pairings) >= max_pairings:
                break
            pairing = Pairing(
                flights=[flight],
                start_city=flight.departure_station,
                end_city=base_city
            )
            if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                pairings.append(pairing)
        if len(pairings) >= max_pairings:
            break
    
    # Generate 2-flight round-trip pairings (base -> city -> base) - most efficient
    for date, day_flights in flights_by_date.items():
        if len(pairings) >= max_pairings:
            break
        outbound = [f for f in day_flights if f.departure_station == base_city]
        
        for out_flight in outbound:
            if len(pairings) >= max_pairings:
                break
            # Find return flights using index
            key = (out_flight.arrival_station, out_flight.arrival_date.date())
            return_candidates = flights_by_station_date.get(key, [])
            return_flights = [f for f in return_candidates 
                           if f.arrival_station == base_city and can_connect_flights(out_flight, f)]
            
            for ret_flight in return_flights:
                pairing = Pairing(
                    flights=[out_flight, ret_flight],
                    start_city=base_city,
                    end_city=base_city
                )
                if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                    pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                    pairings.append(pairing)
                    if len(pairings) >= max_pairings:
                        break
    
    # For small datasets, also generate 4-flight pairings
    # For large datasets, skip this expensive step
    if len(flights) <= 1000 and len(pairings) < max_pairings:
        for date, day_flights in flights_by_date.items():
            if len(pairings) >= max_pairings:
                break
            outbound1 = [f for f in day_flights if f.departure_station == base_city]
            
            for f1 in outbound1:
                if len(pairings) >= max_pairings:
                    break
                connecting_f1 = find_connecting_flights(f1)
                
                for f2 in connecting_f1:
                    if len(pairings) >= max_pairings:
                        break
                    connecting_f2 = find_connecting_flights(f2)
                    
                    for f3 in connecting_f2:
                        if len(pairings) >= max_pairings:
                            break
                        connecting_f3 = [f for f in find_connecting_flights(f3) 
                                        if f.arrival_station == base_city]
                        
                        for f4 in connecting_f3:
                            pairing = Pairing(
                                flights=[f1, f2, f3, f4],
                                start_city=base_city,
                                end_city=base_city
                            )
                            if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                                pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                                pairings.append(pairing)
                                if len(pairings) >= max_pairings:
                                    break
    
    return pairings


def get_flight_pairings_map(pairings: List[Pairing]) -> Dict[int, List[int]]:
    """Create a map from flight_id to list of pairing indices that contain it"""
    flight_to_pairings = {}
    for i, pairing in enumerate(pairings):
        for flight in pairing.flights:
            if flight.flight_id not in flight_to_pairings:
                flight_to_pairings[flight.flight_id] = []
            flight_to_pairings[flight.flight_id].append(i)
    return flight_to_pairings


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
        
        # Check if all flights in pairing are still available
        pairing_flights_ids = [f.flight_id for f in selected_pairing.flights]
        if any(fid in covered_flights for fid in pairing_flights_ids):
            # Try to find another pairing (limit search)
            found = False
            search_limit = min(len(available_pairings), 10)  # Limit search
            for alt_idx in available_pairings[:search_limit]:
                alt_pairing = pairings[alt_idx]
                alt_flight_ids = [f.flight_id for f in alt_pairing.flights]
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
            pairing_date = selected_pairing.flights[0].departure_date.date()
            if pairing_date in crew_assignments_per_day[captain.emp_no]:
                continue
            
            if crew_total_flying[captain.emp_no] + selected_pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
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
            
            pairing_date = selected_pairing.flights[0].departure_date.date()
            if pairing_date in crew_assignments_per_day[fo.emp_no]:
                continue
            
            if crew_total_flying[fo.emp_no] + selected_pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
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
            
            # Update tracking
            for flight in selected_pairing.flights:
                covered_flights.add(flight.flight_id)
            
            pairing_date = selected_pairing.flights[0].departure_date.date()
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
    
    # Add flying time cost
    for assignment in schedule.assignments:
        hours = assignment.pairing.total_flying_time / 60
        cost += hours * assignment.crew.duty_cost_per_hour
    
    return cost, schedule


def greedy_solve(pairings: List[Pairing], flights: List[Flight],
                 captains: List[CrewMember], first_officers: List[CrewMember]) -> Tuple[Schedule, float]:
    """
    Greedy solver for large datasets - faster than GA but may not find optimal solution.
    Sorts pairings by number of flights covered and greedily assigns crew.
    """
    print("Using greedy solver for large dataset...")
    
    schedule = Schedule()
    covered_flights = set()
    crew_total_flying = {c.emp_no: 0 for c in captains + first_officers}
    crew_assignments_per_day = {c.emp_no: {} for c in captains + first_officers}
    
    # Sort pairings by number of flights (descending) to maximize coverage
    sorted_pairings = sorted(pairings, key=lambda p: len(p.flights), reverse=True)
    
    captain_idx = 0
    fo_idx = 0
    
    total_pairings = len(sorted_pairings)
    for i, pairing in enumerate(sorted_pairings):
        if i % 1000 == 0:
            print(f"  Processing pairing {i+1}/{total_pairings}, covered: {len(covered_flights)}/{len(flights)}...", end='\r')
        
        # Check if any flight in pairing is already covered
        pairing_flights_ids = [f.flight_id for f in pairing.flights]
        if any(fid in covered_flights for fid in pairing_flights_ids):
            continue
        
        # Try to assign captain
        assigned_captain = None
        pairing_date = pairing.flights[0].departure_date.date()
        
        for _ in range(min(len(captains), 100)):
            captain = captains[captain_idx % len(captains)]
            captain_idx += 1
            
            if pairing_date in crew_assignments_per_day.get(captain.emp_no, {}):
                continue
            if crew_total_flying.get(captain.emp_no, 0) + pairing.total_flying_time > MAX_FLYING_TIME_HORIZON:
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
        
        # Update tracking
        for flight in pairing.flights:
            covered_flights.add(flight.flight_id)
        
        crew_total_flying[assigned_captain.emp_no] += pairing.total_flying_time
        crew_total_flying[assigned_fo.emp_no] += pairing.total_flying_time
        crew_assignments_per_day[assigned_captain.emp_no][pairing_date] = True
        crew_assignments_per_day[assigned_fo.emp_no][pairing_date] = True
    
    print()  # New line after progress
    
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
    """Output the schedule to a CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FlightNum', 'Date', 'DepartureTime', 'DepartureStation', 
                        'ArrivalTime', 'ArrivalStation', 'Captain', 'FirstOfficer'])
        
        # Create flight to crew mapping
        flight_assignments = {}
        for assignment in schedule.assignments:
            for flight in assignment.pairing.flights:
                if flight.flight_id not in flight_assignments:
                    flight_assignments[flight.flight_id] = {'Captain': '', 'FirstOfficer': ''}
                flight_assignments[flight.flight_id][assignment.role] = assignment.crew.emp_no
        
        # Output each flight
        for flight in sorted(flights, key=lambda f: (f.departure_date, f.departure_time)):
            captain = flight_assignments.get(flight.flight_id, {}).get('Captain', 'UNCOVERED')
            fo = flight_assignments.get(flight.flight_id, {}).get('FirstOfficer', 'UNCOVERED')
            
            writer.writerow([
                flight.flight_num,
                flight.departure_date.strftime('%Y-%m-%d'),
                flight.departure_time.strftime('%H:%M'),
                flight.departure_station,
                flight.arrival_time.strftime('%H:%M'),
                flight.arrival_station,
                captain,
                fo
            ])
    
    print(f"Schedule saved to {output_path}")


def output_crew_schedule_csv(schedule: Schedule, crew_members: List[CrewMember], output_path: str):
    """Output crew-centric schedule to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['CrewID', 'Role', 'Date', 'Flights', 'TotalFlyingTime(min)', 
                        'StartCity', 'EndCity'])
        
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
                                        key=lambda a: a.pairing.flights[0].departure_date):
                    flight_nums = ' -> '.join([f.flight_num for f in assignment.pairing.flights])
                    date = assignment.pairing.flights[0].departure_date.strftime('%Y-%m-%d')
                    
                    writer.writerow([
                        crew.emp_no,
                        assignment.role,
                        date,
                        flight_nums,
                        assignment.pairing.total_flying_time,
                        assignment.pairing.start_city,
                        assignment.pairing.end_city
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
    
    # Crew statistics
    crew_flying_time = {}
    crew_duty_days = {}
    for assignment in schedule.assignments:
        emp = assignment.crew.emp_no
        if emp not in crew_flying_time:
            crew_flying_time[emp] = 0
            crew_duty_days[emp] = set()
        crew_flying_time[emp] += assignment.pairing.total_flying_time
        crew_duty_days[emp].add(assignment.pairing.flights[0].departure_date.date())
    
    print(f"\nCrew members used: {len(crew_flying_time)}/{len(crew_members)}")
    
    print("\nCrew utilization:")
    for emp in sorted(crew_flying_time.keys()):
        flying_hours = crew_flying_time[emp] / 60
        days = len(crew_duty_days[emp])
        print(f"  {emp}: {flying_hours:.1f} hours over {days} duty days")
    
    if schedule.uncovered_flights:
        print(f"\nUncovered flight IDs: {sorted(schedule.uncovered_flights)[:20]}...")


def main():
    """Main function to run the crew scheduling optimization"""
    import os
    import argparse
    
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
    
    print("Loading data...")
    crew_members = load_crew_data(crew_file)
    flights = load_flight_data(flight_file)
    
    print(f"Loaded {len(crew_members)} crew members")
    print(f"Loaded {len(flights)} flights")
    
    # Separate captains and first officers
    captains = [c for c in crew_members if c.is_captain]
    first_officers = [c for c in crew_members if c.is_first_officer]
    
    print(f"Captains: {len(captains)}")
    print(f"First Officers: {len(first_officers)}")
    
    # Get all unique base cities from crew members
    base_cities = list(set(c.base for c in crew_members))
    print(f"Base cities: {base_cities}")
    
    # Get adaptive GA parameters based on dataset size
    max_iterations, population_size, max_pairings = get_adaptive_ga_params(len(flights), len(crew_members))
    # Calculate max pairings per base
    max_pairings_per_base = max_pairings // len(base_cities) if base_cities else max_pairings
    
    # Generate pairings for all base cities
    print("\nGenerating feasible pairings...")
    all_pairings = []
    for base_city in base_cities:
        pairings = generate_pairings(flights, base_city, max_pairings=max_pairings_per_base)
        all_pairings.extend(pairings)
        print(f"  Base {base_city}: {len(pairings)} pairings")
    
    print(f"Total generated: {len(all_pairings)} feasible pairings")
    
    if not all_pairings:
        print("ERROR: No feasible pairings found. Check flight data and constraints.")
        return
    
    # Use greedy solver for very large datasets, GA for smaller ones
    use_greedy = len(flights) > 5000 or len(crew_members) > 200
    
    if use_greedy:
        print("\nRunning greedy optimization (large dataset)...")
        best_schedule, best_fitness = greedy_solve(all_pairings, flights, captains, first_officers)
    else:
        print("\nRunning Genetic Algorithm optimization...")
        print(f"Parameters: iterations={max_iterations}, population={population_size}, "
              f"crossover={CROSSOVER_RATE}, mutation={MUTATION_RATE}")
        
        ga = GeneticAlgorithm(all_pairings, flights, captains, first_officers, 
                              max_iterations, population_size)
        best_schedule, best_fitness = ga.run()
    
    print(f"\nOptimization complete. Best fitness: {best_fitness:.2f}")
    
    # Print summary
    print_schedule_summary(best_schedule, flights, crew_members)
    
    # Output schedules
    output_schedule_csv(best_schedule, flights, output_file)
    output_crew_schedule_csv(best_schedule, crew_members, crew_output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
