# Problem Formulation

In this section, a new mathematical model is presented for CSP. Sets, indices, parameters, and decision variables are as follows:

## Sets and Indices

- $I$: Set of all crew members
- $i$: Index of crew member ($i \in I$)
- $J$: Set of all cities
- $j$: Index of city ($j \in J$)
- $D$: Set of all days of planning horizon (Saturday to Friday)
- $d$: Index of day ($d \in D$)
- $F$: Set of all duties
- $f$: Index of duty ($f \in F$)
- $N$: Set of flight rounds. Suppose that a crew member starts his/her duty from city $A$ to $B$ and continues it by flight from $B$ to $C$. For the flight $(A, B)$, $n$ equals one and for the flight $(B, C)$, $n$ equals two
- $n$: Index of flight round ($n \in N$)
- $o_i$: Home base of crew member $i$

## Parameters

- $de_{jj'd}$: Departure time of flight $jj'$ (flight from city $j$ to $j'$) on day $d$
- $l_{jj'd}$: Arrival time of flight $jj'$ on day $d$
- $min\_sit$: Minimum required time between two consecutive flights on a single duty day
- $min\_rest$: Minimum rest required time between two consecutive flights on consecutive days
- $upper1$: Maximum flying time per duty
- $upper2$: Maximum flying time in the planning horizon
- $upper3$: Maximum elapsed time per duty
- $e_{jj'd}$: Deadhead cost of flight $jj'$ on day $d$
- $k_{jj'd}$: The cost of uncovered flight $jj'$ on day $d$
- $h$: Hoteling cost of crew member
- $r_i$: Cost of using crew $i$
- $M$: A big number

## Decision Variables

- $x_{ijj'dn}$: 1 if the flight $jj'$ on day $d$ is assigned to crew member $i$ in flight round $n$, 0 otherwise
- $z_{ijj''dn}$: 1 if flight $j'j''$ is handled after flight $jj'$ on day $d$ by crew member $i$, 0 otherwise
- $s_{ijd}$: 1 if crew member $i$ starts his/her duty on day $d$ from city $j$, 0 otherwise
- $w_{ijd}$: 1 if crew member $i$ ends his/her duty on day $d$ to city $j$, 0 otherwise
- $y_{idf}$: 1 if crew member $i$ handles his/her $f$-th duty on day $d$, 0 otherwise ($f \leq d$)
- $ss_i$: 1 if crew member $i$ is assigned a duty in planning horizon, 0 otherwise
- $v_{jj'd}$: 1 if flight $jj'$ on day $d$ is covered, 0 otherwise
- $b_{jj'd}$: Integer variable indicating the number of extra times that flight $jj'$ is covered on day $d$ (concept of deadhead)

## Objective Function and Constraints

The objective function tends to minimize the total cost of the system, including uncovered flights cost, deadhead cost, hoteling cost, and crew cost:

$$
\begin{aligned}
min =& \sum_{d\in D}\sum_{j'\in J}\sum_{j\in J} (1-v_{jj'd}) \cdot k_{jj'd} + \sum_{d\in D}\sum_{j'\in J}\sum_{j\in J} (b_{jj'd}-v_{jj'd}) \cdot e_{jj'd} \\
&+ \sum_{d\in D}\sum_{j\neq o_i}\sum_{i\in I} w_{ijd} \cdot h + \sum_{i\in I} ss_i \cdot r_i
\end{aligned}
$$

### Constraints

1. Each flight can be done in a single flight round or uncovered:

$$
\sum_{n \in N} x_{ijj'dn} \leq 1 \quad \forall i \in I, j, j' \in J, d \in D
$$

2. Each crew member can handle at most one flight in a flight round:

$$
\sum_{d\in D} \sum_{j'\in J} \sum_{j\in J} x_{ijj'dn} \leq 1 \quad \forall i\in I, n\in N
$$

3. Minimum sit time between two consecutive flights on a duty:

$$
de_{j'j''d} - l_{jj'd} \geq min\_sit - M \cdot (2 - x_{ij'j''d,n+1} - x_{ijj'dn}) \quad \forall i\in I, j,j',j''\in J, d\in D, n\in N
$$

4. Minimum rest time between two consecutive flights on consecutive days:

$$
1440 + de_{j'j''d+1} - l_{jj'd} \geq min\_rest - M \cdot (2 - x_{ij'j''d+1,n+1} - x_{ijj'dn}) \quad \forall i\in I, j,j',j''\in J, d\in D, n\in N
$$

5. Each crew member must start his/her first duty from home base:

$$
s_{io_id} \geq -M \cdot (1 - y_{idf}) + 1 \quad \forall i\in I, j\in o_i, f \in F, f=1
$$

6. Each crew member must come back to home base at the end of his/her last duty:

$$
w_{io_id} \geq 1 - M \cdot (1 - y_{idf}) - M \cdot \left( \sum_{\substack{d'=d+1 \\ d'\in D}} \sum_{\substack{f'=f+1 \\ f'\in F}} y_{id'f'} \right) \quad \forall i\in I, j\in o_i, d\in D, f\in F: f\leq d
$$

7. Integrity of consecutive duties (end city of previous duty = start city of next duty):

$$
s_{ij'd'} \geq w_{ijd} - M \cdot (2 - y_{idf} - y_{id'f+1}) \quad \forall i\in I, j\in J, d,d'\in D: d' > d
$$

$$
s_{ij'd'} \leq w_{ijd} + M \cdot (2 - y_{idf} - y_{id'f+1}) \quad \forall i\in I, j\in J, d,d'\in D: d' > d
$$

8. Flight integrity on a duty day:

$$
s_{ijd} + \sum_{n \in N} \sum_{j' \in J} x_{ij'jd} = \sum_{n \in N} \sum_{j' \in J} x_{ijj'd} + w_{ijd} \quad \forall i \in I, j \in J, d \in D
$$

9. No flight assigned to crew on nonworking day:

$$
\sum_{n\in N}\sum_{j'\in J}\sum_{j\in J} x_{ijj'dn} \geq -M \cdot (1 - \sum_{f\in F} y_{idf}) + \varepsilon \quad \forall i\in I, d\in D
$$

$$
\sum_{n \in N} \sum_{j' \in J} \sum_{j \in J} x_{ijj'dn} \leq M \cdot \sum_{f \in F} y_{idf} \quad \forall i \in I, d \in D
$$

10. Each crew member can handle at most one duty per day:

$$
\sum_{f \in F} y_{idf} \leq 1 \quad \forall i \in I, d \in D
$$

11. Each duty can be assigned to a crew member at most once:

$$
\sum_{d \in D} y_{idf} \leq 1 \quad \forall i \in I, f \in F
$$

12. Duty index cannot be less than day index:

$$
y_{idf} = 0 \quad \forall i \in I, d\in D, f \in F, d < f
$$

13. Crew cost is considered if duty is assigned:

$$
M \cdot ss_i \geq \sum_{d \in D} \sum_{f \in F} y_{idf} \quad \forall i \in I
$$

14. Flight origin constraint for first duty:

$$
1 - M \cdot (2 - y_{idf} - x_{ijj'd,n+1}) \leq \sum_{j'' \in J} x_{ij''jd} \quad \forall i\in I, j,j'\in J, d\in D, n\in N, f\in F, f=1
$$

$$
\sum_{j''\in J} x_{ij''jd} \leq 1 + M \cdot (2 - y_{idf} - x_{ijj'd,n+1}) \quad \forall i\in I, j,j'\in J, d\in D, n\in N, f\in F, f=1
$$

15. Flight origin constraint for subsequent duties:

$$
1 - M \cdot (3 - y_{id'f+1} - y_{idf} - x_{ijj'd',n+1}) \leq \sum_{j''\in J} x_{ij''jd'n} + \sum_{j''\in J} x_{ij''jd'n} \quad \forall i\in I, j,j'\in J, d,d'\in D: d' > d, f\in F, n\in N
$$

$$
\sum_{j''\in J} x_{ij''jd'n} + \sum_{j''\in J} x_{ij''jd'n} \leq 1 + M \cdot (3 - y_{id'f+1} - y_{idf} - x_{ijj'd',n+1}) \quad \forall i\in I, j,j'\in J, d,d'\in D: d' > d, f\in F, n\in N
$$

16. First flight starts from home base:

$$
\sum_{j'\in J} x_{ijj'dn} \geq s_{ijd} - M \cdot (1 - y_{idf}) \quad \forall i\in I, j\in J, d\in D, f\in F, f=1, n\in N, n=1
$$

17. Integrity between two consecutive duties:

$$
\begin{aligned}
\sum_{j' \in J} x_{ijj'dn} \geq & s_{ijd} - M \cdot (2 - y_{id'f} - y_{idf+1}) - M \cdot \left( 1 - \sum_{j''\in J} \sum_{j'''\in J} x_{ij''j'''d'n-1} \right) \\
& - M \cdot \left( \sum_{j''\in J} \sum_{j'''''\in J} x_{ij''j''''d'n} \right) \quad \forall i \in I, j \in J, d, d' \in D: d' < d, n \in N
\end{aligned}
$$

18. Working and nonworking days constraint:

$$
\sum_{d\in D}\sum_{j\in J} s_{ijd} \leq (d'' - d' + 1) + M \cdot (2 - y_{id'f} - y_{id''f'}) + M \cdot \left( \sum_{d'''\in D} y_{id'''(f'+1)} \right) \quad \forall i\in I, d'',d'\in D, f',f\in F, f=1
$$

19. Crew is not used if no duty is assigned:

$$
\sum_{j\in J} s_{ijd} \leq M \cdot \left( \sum_{d\in D}\sum_{f\in F} y_{idf} \right) \quad \forall i\in I
$$

20. Relationship between consecutive duties:

$$
\sum_{\substack{d' < d \\ d'\in D}} y_{id'f} \leq 1 + M \cdot (1 - y_{idf+1}) \quad \forall i\in I, d\in D, f\in F, f+1 \leq d
$$

$$
1 - M \cdot (1 - y_{idf+1}) \leq \sum_{\substack{d' < d \\ d'\in D}} y_{id'f} \quad \forall i\in I, d\in D, f\in F, f+1 \leq d
$$

21. Flight coverage constraint:

$$
\sum_{n\in N} \sum_{i\in I} x_{ijj'dn} \geq v_{jj'd} \quad \forall j,j'\in J, d\in D
$$

$$
\sum_{n\in N} \sum_{i\in I} x_{ijj'dn} \leq M \cdot v_{jj'd} \quad \forall j,j'\in J, d\in D
$$

22. Deadhead flight constraint:

$$
b_{jj'd} \leq M \cdot v_{jj'd} \quad \forall j,j'\in J, d\in D
$$

$$
b_{jj'd} \geq \sum_{n} \sum_{i} x_{ijj'dn} - M \cdot (1 - v_{jj'd}) \quad \forall j,j'\in J, d\in D
$$

23. Maximum flying time per duty:

$$
\sum_{n\in N} \sum_{j'\in J}\sum_{j\in J} (l_{jj'd} - de_{jj'd}) \cdot x_{ijj'dn} \leq upper1 \quad \forall i\in I, d\in D
$$

24. Maximum flying time in planning horizon:

$$
\sum_{n \in N} \sum_{d \in D} \sum_{j' \in J} \sum_{j \in J} (l_{jj'd} - de_{jj'd}) \cdot x_{ijj'dn} \leq upper2 \quad \forall i \in I
$$

25. Flight sequence constraint:

$$
z_{ijj''dn} \geq x_{ijj'dn} + x_{ij'j''d,n+1} - 1 \quad \forall i\in I, j,j',j''\in J, d\in D, n\in N
$$

26. Maximum elapsed time per duty:

$$
\begin{aligned}
\sum_{n\in N}\sum_{j'\in J}\sum_{j\in J} (l_{jj'd} - de_{jj'd}) \cdot x_{ijj'dn} + \sum_{n\in N}\sum_{j\in J}\sum_{j'\in J}\sum_{j''\in J} (de_{j'j''d} - l_{jj'd}) \cdot z_{ijj''dn} \leq upper3 \quad \forall i\in I, d\in D
\end{aligned}
$$

# Problem Solving Approach

As CSP is an NP-hard problem, it is not possible to achieve an optimal solution for large-scale problem. As a result, two metaheuristic-based approaches, namely, Particle Swarm Optimization (PSO) and Genetic Algorithms (GA), have been used to find a near-optimal solution over a reasonable time. In this section, the applied metaheuristic algorithms are presented.

## Genetic Algorithm

GA is the most well-known type of evolutionary algorithms developed by \citet{Holland1992}. This algorithm is population-based, and the initial solutions are produced through the proper encoding of the problem. Then, each of the solutions is evaluated using a fitness function. In the next step, some solutions (parents) are selected through the Roulette Wheel Selection mechanism and create the Mating Pool. Subsequently, off-springs are generated through crossover and mutation of parents. By adding the off-springs to the set of solutions, the possibility of reaching better solutions increases. This procedure continues as long as the termination conditions are prepared.

### Chromosome Encoding

In this study, all possible pairings are created according to the existing crew base with two and four flights. It should be noted that all the FAA rules are considered during creation. Figure 1 shows the chromosome encoding that is used to represent the problem solutions. In this paper, this chromosome is a row vector that its columns are equal to the number of available flights (gens). These columns are sorted in a descending order based on the number of pairings related to flights. In other words, the first columns show the flights with less pairings, and the last ones show the flights with more pairings. Each column is filled by a random key number. Random keys were introduced by \citet{Bean1994} for the first time. The numbers have the uniform distribution between 0 and 1. The flight selection starts from the beginning of this vector and continues till the end.

**Table: Problem encoding (Figure 1)**

| Chromosome | 0.53 | 0.43 | 0.76 | 0.01 | 0.32 | 0.87 | 0.12 | 0.25 | 0.74 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

After selecting the flight, the pairing that contains that flight is determined through the chromosome. In fact, the random number in this chromosome is multiplied by the number of pairings related to the selected flight, and the desirable pairing is determined. The above-mentioned procedure is repeated for other selections. During flight selections, if a flight was selected by a pairing(s) before, it will be neglected. The selection procedure continues until no flight is unselected. In this study, a multipoint cross-over operator is used, and the mechanism of mutation is to select a number of genes randomly and replace them with new random numbers.

### Pseudocode of GA

```text
Begin
    Initialize population with random candidate solutions
    Evaluate each chromosome according to the fitness function
    Repeat until the termination condition is satisfied
        Select parents
        Apply cross-over operator
        Apply mutation operator
        Evaluate new candidates
        Select individuals for the next generation
    End
```

## Particle Swarm Optimization

PSO algorithm is an evolutionary algorithm, which is represented based on the social behavior of birds. Kennedy and Eberhart introduced this algorithm in 1995. This algorithm is a population-based algorithm, and the change in each individual's position (particle) is based on its previous movements and the experience of neighboring particles. If a particle can get better situations in these replacements, these experiences will be stored in the particle's memory and will affect the next changes in the position of the particle. This process continues to the point that achieving a better solution is not possible.

Consider that the Swarm size is equal to $N$, and the position related to each of the particles in the repetition $t$ is in the form of $X_i(t) = (x_{i1}(t), x_{i2}(t), ..., x_{iD}(t))$ ($i = 1,2,...,N$). For each solution, the vector called the velocity vector is attributed to $V_i(t) = (v_{i1}(t), v_{i2}(t), ..., v_{iD}(t))$. The best solution obtained from the specified swarm until $t$ repetition is in the form of $P_i(t) = (p_{i1}(t), p_{i2}(t), ..., p_{iD}(t))$, and the best solution obtained from the particles, until this repetition, is $P_g(t) = (p_{g1}(t), p_{g2}(t), ..., p_{gD}(t))$ \citep{Eddaly2016}. The velocity vector and position of each swarm, in the repetition of $t+1$, are adjustable according to the following formulas:

$$
v_{ij}(t+1) = w \cdot v_{ij}(t) + c_1 \cdot r_1 \cdot (p_{ij}(t) - x_{ij}(t)) + c_2 \cdot r_2 \cdot (p_{gj}(t) - x_{ij}(t))
$$

$$
x_{ij}(t+1) = x_{ij}(t) + v_{ij}(t+1)
$$

where $c_1$ is cognition learning factor, $c_2$ social learning factor, and $r_1, r_2$ random numbers with uniform probability distribution between 0 and 1. $w$ is the inertia weight parameter that affects the previous velocity of particle on its current velocity.

### Pseudocode of PSO

```text
Begin
    Initialize population with random candidate solutions
    Evaluate each chromosome according to the fitness function
    Step 1: Define the search space, population size, and objective function
    Step 2: Initialize a swarm of particles with random positions and velocities in the problem space
    Step 3: Define the local solution and the global best solution
    Step 4: Update the current positions and velocities using Eq (36) and Eq (37)
    Step 5: Calculate the value of the objective function for each particle and update the current local best and global best
    Step 6: Go back to step (4) until a stop criterion is satisfied, usually a sufficiently good fitness or a specified number of iterations
End
```

## Parameters Setting

The efficiency of the proposed algorithms considerably depends on the applied parameters. Taguchi first presented the parameter design in early 1960s. The main GA parameters/factors are maximum iteration, initial population, crossover rate, and mutation rate, while in PSO maximum iteration, Swarm size, the inertia weight ($w$), cognitive learning factor ($c_1$), and social learning factor ($c_2$) are considered as the main parameters. Each of the parameters mentioned above is valued in 3 levels.

### For GA

- Maximum iteration: 80, 100, 120
- Initial population: 100, 150, 200
- Crossover rate: 0.5, 0.6, 0.7
- Mutation rate: 0.1, 0.2, 0.3

### For PSO

- Maximum iteration: 60, 80, 100
- Swarm size: 100, 140, 180
- Inertia weight ($w$): 0.5, 0.6, 0.7
- Cognitive learning factor ($c_1$): 1.9, 2.0, 2.1
- Social learning factor ($c_2$): 1.9, 2.0, 2.1

For GA, 9 different scenarios are considered, while 27 different scenarios are assumed for PSO. Each scenario is repeated five times, and the mean S/N ratio plots for the GA and the PSO objective functions are depicted in Figures 4 and 5. Finally, the optimal values of these parameters are obtained as follows:

- GA Parameters: max iteration = 120, Population size = 200, crossover rate = 0.6, mutation rate = 0.3
- PSO Parameters: max iteration = 100, swarm size = 180, $w = 0.7$, $c_1 = 2.1$, $c_2 = 1.9$
