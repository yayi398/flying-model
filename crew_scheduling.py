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

# GA Parameters (from README.md)
MAX_ITERATIONS = 150
POPULATION_SIZE = 250
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.25

# Cost parameters
UNCOVERED_FLIGHT_COST = 10000  # High penalty for uncovered flights
DEADHEAD_COST = 500  # Cost for deadhead flights
HOTEL_COST = 200  # Cost for crew staying overnight away from base
CREW_USAGE_COST = 100  # Base cost for using a crew member


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
            crew = CrewMember(
                emp_no=row['EmpNo'],
                is_captain=row['Captain'] == 'Y',
                is_first_officer=row['FirstOfficer'] == 'Y',
                can_deadhead=row['Deadhead'] == 'Y',
                base=row['Base'],
                duty_cost_per_hour=float(row['DutyCostPerHour']),
                pairing_cost_per_hour=float(row.get('PairingCostPerHour', row.get('ParingCostPerHour', '0')))
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


def generate_pairings(flights: List[Flight], base_city: str, max_flights: int = 4) -> List[Pairing]:
    """
    Generate all feasible pairings starting and ending at the base city.
    A pairing can have 1-4 flights following FAA rules.
    Also handles special cases where flights start/end at base city.
    """
    pairings = []
    
    # Group flights by date
    flights_by_date = {}
    for flight in flights:
        date_key = flight.departure_date.date()
        if date_key not in flights_by_date:
            flights_by_date[date_key] = []
        flights_by_date[date_key].append(flight)
    
    # Generate single-flight pairings for flights that end at base (deadhead home)
    for date, day_flights in flights_by_date.items():
        # Flights that arrive at base (crew can deadhead out, fly back)
        inbound_only = [f for f in day_flights if f.arrival_station == base_city]
        for flight in inbound_only:
            pairing = Pairing(
                flights=[flight],
                start_city=flight.departure_station,  # Deadhead to departure
                end_city=base_city
            )
            if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                pairings.append(pairing)
    
    # Generate single round-trip pairings (2 flights: base -> city -> base)
    for date, day_flights in flights_by_date.items():
        outbound = [f for f in day_flights if f.departure_station == base_city]
        
        for out_flight in outbound:
            # Find return flights
            return_flights = [f for f in day_flights 
                           if f.departure_station == out_flight.arrival_station 
                           and f.arrival_station == base_city]
            
            for ret_flight in return_flights:
                if can_connect_flights(out_flight, ret_flight):
                    pairing = Pairing(
                        flights=[out_flight, ret_flight],
                        start_city=base_city,
                        end_city=base_city
                    )
                    # Validate pairing constraints
                    if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                        pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                        pairings.append(pairing)
    
    # Generate 3-flight pairings to cover orphan return flights 
    # (e.g., outbound + 2 return flights from same intermediate city)
    for date, day_flights in flights_by_date.items():
        outbound = [f for f in day_flights if f.departure_station == base_city]
        
        for f1 in outbound:
            # f1: base -> city_A
            return_from_A = [f for f in day_flights 
                           if f.departure_station == f1.arrival_station 
                           and f.arrival_station == base_city
                           and can_connect_flights(f1, f)]
            
            # Look for 3-flight patterns: base -> A, A -> base, base -> B -> base
            for f2 in return_from_A:
                outbound_after = [f for f in day_flights
                                 if f.departure_station == base_city
                                 and can_connect_flights(f2, f)]
                
                for f3 in outbound_after:
                    return_f3 = [f for f in day_flights
                                if f.departure_station == f3.arrival_station
                                and f.arrival_station == base_city
                                and can_connect_flights(f3, f)]
                    
                    for f4 in return_f3:
                        pairing = Pairing(
                            flights=[f1, f2, f3, f4],
                            start_city=base_city,
                            end_city=base_city
                        )
                        if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                            pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                            pairings.append(pairing)
    
    # Generate 4-flight pairings (base -> A -> base -> B -> base or base -> A -> B -> A -> base)
    for date, day_flights in flights_by_date.items():
        outbound1 = [f for f in day_flights if f.departure_station == base_city]
        
        for f1 in outbound1:
            # f1: base -> city_A
            inbound1 = [f for f in day_flights 
                       if f.departure_station == f1.arrival_station 
                       and can_connect_flights(f1, f)]
            
            for f2 in inbound1:
                # f2: city_A -> city_B (or back to base)
                outbound2 = [f for f in day_flights
                            if f.departure_station == f2.arrival_station
                            and can_connect_flights(f2, f)]
                
                for f3 in outbound2:
                    # f3: city_B -> city_C
                    inbound2 = [f for f in day_flights
                               if f.departure_station == f3.arrival_station
                               and f.arrival_station == base_city
                               and can_connect_flights(f3, f)]
                    
                    for f4 in inbound2:
                        # f4: city_C -> base
                        pairing = Pairing(
                            flights=[f1, f2, f3, f4],
                            start_city=base_city,
                            end_city=base_city
                        )
                        if (pairing.total_flying_time <= MAX_FLYING_TIME_PER_DUTY and
                            pairing.elapsed_time <= MAX_ELAPSED_TIME_PER_DUTY):
                            pairings.append(pairing)
    
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
    """
    schedule = Schedule()
    covered_flights = set()
    crew_total_flying = {c.emp_no: 0 for c in captains + first_officers}
    crew_assignments_per_day = {c.emp_no: {} for c in captains + first_officers}
    
    # Sort flights by number of available pairings (ascending) - prioritize harder flights
    flight_order = sorted(range(len(flights)), 
                         key=lambda fid: len(flight_to_pairings.get(fid, [])))
    
    # Assign pairings based on chromosome
    captain_idx = 0
    fo_idx = 0
    
    for flight_idx in flight_order:
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
            # Try to find another pairing
            found = False
            for alt_idx in available_pairings:
                alt_pairing = pairings[alt_idx]
                alt_flight_ids = [f.flight_id for f in alt_pairing.flights]
                if not any(fid in covered_flights for fid in alt_flight_ids):
                    selected_pairing = alt_pairing
                    found = True
                    break
            if not found:
                continue
        
        # Assign captain
        assigned_captain = None
        for _ in range(len(captains)):
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
        
        # Assign first officer
        assigned_fo = None
        for _ in range(len(first_officers)):
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


class GeneticAlgorithm:
    """Genetic Algorithm for Crew Scheduling Problem"""
    
    def __init__(self, pairings: List[Pairing], flights: List[Flight],
                 captains: List[CrewMember], first_officers: List[CrewMember]):
        self.pairings = pairings
        self.flights = flights
        self.captains = captains
        self.first_officers = first_officers
        self.flight_to_pairings = get_flight_pairings_map(pairings)
        self.chromosome_length = len(flights)
        
    def initialize_population(self) -> List[List[float]]:
        """Initialize population with random chromosomes"""
        population = []
        for _ in range(POPULATION_SIZE):
            chromosome = [random.random() for _ in range(self.chromosome_length)]
            population.append(chromosome)
        return population
    
    def evaluate_population(self, population: List[List[float]]) -> List[Tuple[float, Schedule, List[float]]]:
        """Evaluate fitness of all chromosomes"""
        results = []
        for chromosome in population:
            fitness, schedule = calculate_fitness(
                chromosome, self.pairings, self.flights,
                self.flight_to_pairings, self.captains, self.first_officers
            )
            results.append((fitness, schedule, chromosome))
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
        
        for iteration in range(MAX_ITERATIONS):
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
            elite_size = max(1, POPULATION_SIZE // 10)
            for i in range(elite_size):
                new_population.append(evaluated_pop[i][2])
            
            # Fill rest with offspring
            while len(new_population) < POPULATION_SIZE:
                parent1 = self.roulette_wheel_selection(evaluated_pop)
                parent2 = self.roulette_wheel_selection(evaluated_pop)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < POPULATION_SIZE:
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
    
    # Get base city (all crew have same base in this dataset)
    base_city = crew_members[0].base
    print(f"Base city: {base_city}")
    
    print("\nGenerating feasible pairings...")
    pairings = generate_pairings(flights, base_city)
    print(f"Generated {len(pairings)} feasible pairings")
    
    if not pairings:
        print("ERROR: No feasible pairings found. Check flight data and constraints.")
        return
    
    print("\nRunning Genetic Algorithm optimization...")
    print(f"Parameters: iterations={MAX_ITERATIONS}, population={POPULATION_SIZE}, "
          f"crossover={CROSSOVER_RATE}, mutation={MUTATION_RATE}")
    
    ga = GeneticAlgorithm(pairings, flights, captains, first_officers)
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
