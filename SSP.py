import math
import random
import argparse
import csv

# ------------- Defining helper functions ------------- 

# Load problem from CSV file
def load_problem_from_csv(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        target = int(rows[0][0])
        numbers = [int(x.strip()) for x in rows[1]]
        return numbers, target

# Defining our objective function, the closer it is to 0 the better
def objective_function(solution_vector, numbers, target_sum):
    total = sum([vec * num for vec,num in zip(solution_vector,numbers)])
    return abs(target_sum - total)

# Changes one bit of a solution vector
def get_neighbors(solution_vector):
    neighbors = []
    for i in range(len(solution_vector)):
        neighbor = solution_vector.copy()
        neighbor[i] = 1 - neighbor[i] # bit flip
        neighbors.append(neighbor)
    return neighbors

# Generates random solution vector
def generate_random_solution(n):
    return [random.randint(0,1) for _ in range(n)]

# Generates random number vector
def generate_random_number_vector(rand_min,rand_max,n):
    return [random.randint(rand_min, rand_max) for _ in range(n)]

# ------------- Defining algorithms ------------- 

# Brute force algorithm
def brute_force_algorithm(numbers_array ,target):
    print('Brute force algorithm:')
    print(f'Numbers array = {numbers_array}')
    print(f'Target that we\'re looking for = {target}')
    n = len(numbers_array)
    best_solution_vector = []
    best_solution = float('inf')

    for mask in range(2 ** n): # All combinations are 2 to the N in this case ( 0 to 2^n -1)
        solution_vector = []
        for i in range(n):      # Generating binary vector that represents the mask i.e. mask 7 = [0,1,1,1]
            if ( math.floor(mask / (2**i))) % 2 == 1: # If mask divided to floor by 2^i mod 2 then append
                solution_vector.append(1)
            else:
                solution_vector.append(0)
        
        score = objective_function(solution_vector, numbers_array, target)

        if score < best_solution:
            best_solution = score
            best_solution_vector = solution_vector

        if best_solution == 0: # If it finds best solution then stop searching
            break
    print(f'Solution vector = {best_solution_vector}')
    print(f'Score = {best_solution}')
    return best_solution_vector, best_solution

# In our case of a Subset Sum Problem we'll be looking at neighbors as one bitflip in our solution array.
# Each iteration will have new n-1 neighbors and iterate as long as the score goes closer to 0. It stops when score cannot get better.
def hill_climbing_deterministic_algorithm(numbers_array , target, max_iterations = 1000):
    print('Hill climb algorithm - deterministic method:')
    n = len(numbers_array)
    current_solution = generate_random_solution(n)
    current_score = objective_function(current_solution, numbers_array, target)

    print(f'Numbers array = {numbers_array}')
    print(f'Target that we\'re looking for = {target}')
    print(f'Starting solution = {current_solution}')
    print(f'Starting score  = {current_score}')

    for i in range(max_iterations):
        print(f'Iteration {i}')

        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_score = current_score

        for neighbor in neighbors:
            score = objective_function(neighbor, numbers_array, target)
            if score < best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor

        if best_neighbor_score < current_score and best_neighbor is not None:
            current_solution = best_neighbor
            current_score = best_neighbor_score
        else:
            break

        if current_score == 0:
            break

    print("\nFinal solution:")
    subset = [numbers_array[i] for i in range(n) if current_solution[i] == 1]
    print(f"Subset: {subset}")
    print(f"Sum: {sum(subset)}")
    print(f"Difference from target: {abs(sum(subset) - target)}")

    return current_solution, current_score

def hill_climbing_random_neighbor_algorithm(numbers_array ,target, max_iterations = 1000):
    print('Hill climb algorithm - random neighbor method:')
    n = len(numbers_array)
    current_solution = generate_random_solution(n)
    current_score = objective_function(current_solution, numbers_array, target)

    print(f'Numbers array = {numbers_array}')
    print(f'Target that we\'re looking for = {target}')
    print(f'Starting solution = {current_solution}')
    print(f'Starting score  = {current_score}')

    for i in range(max_iterations):
        print(f'Iteration {i}')

        neighbors = get_neighbors(current_solution)
        better_neighbors = []

        for neighbor in neighbors:
            score = objective_function(neighbor, numbers_array, target)
            if score < current_score:
                better_neighbors.append(neighbor)

        if not better_neighbors:
            break
        else:
            current_solution = random.choice(better_neighbors)
            current_score = objective_function(current_solution, numbers_array, target)

        if current_score == 0:
            break

    print("\nFinal solution:")
    subset = [numbers_array[i] for i in range(n) if current_solution[i] == 1]
    print(f"Subset: {subset}")
    print(f"Sum: {sum(subset)}")
    print(f"Difference from target: {abs(sum(subset) - target)}")

    return current_solution, current_score

def tabu_search_algorithm(numbers_array ,target, max_iterations = 1000, tabu_tenure = None): #tenure means how many iterations we keep the tabu list
    print('Tabu search algorithm:')
    n = len(numbers_array)
    current_solution = generate_random_solution(n)
    current_score = objective_function(current_solution, numbers_array, target)
    best_solution = current_solution.copy()
    best_score = current_score
    tabu_list = []
    tabu_set = set()
    backtrack_stack = [(current_solution.copy(), current_score)]

    print(f'Numbers array = {numbers_array}')
    print(f'Target = {target}')
    print(f'Starting solution = {current_solution}')
    print(f'Starting score = {current_score}')

    for i in range(max_iterations):
        neighbors = get_neighbors(current_solution)
        
        valid_neighbors = []

        # Checking all available neighbors
        for neighbor in neighbors:
            key = str(neighbor)
            score = objective_function(neighbor, numbers_array, target)

            if key not in tabu_set or score < best_score:
                valid_neighbors.append((neighbor, score))
        
        # If there are no neighbors we backtract in the move stack
        if not valid_neighbors:
            if backtrack_stack:
                print(f'[{i}] No valid moves. Backtracking...')
                current_solution, current_score = backtrack_stack.pop()
                continue
            else:
                print(f'[{i}] No valid moves and no backup. Stopping.')
                break

        # We choose the best neighbor
        valid_neighbors.sort(key=lambda x: x[1])  # sort by score
        best_neighbor, best_neighbor_score = valid_neighbors[0]

        # We save the current point if it's better
        if best_neighbor_score < current_score:
            backtrack_stack.append((best_neighbor.copy(), best_neighbor_score))

        # updating tabu list
        key = str(best_neighbor)
        tabu_list.append(key)
        tabu_set.add(key)
        if tabu_tenure is not None and len(tabu_list) > tabu_tenure:
            removed = tabu_list.pop(0)
            tabu_set.remove(removed)

        # updating the current solution
        current_solution = best_neighbor
        current_score = best_neighbor_score

        if current_score < best_score:
            best_solution = current_solution.copy()
            best_score = current_score

        print(f"[{i}] Score: {current_score}, Best: {best_score}, Solution: {current_solution}")

        if best_score == 0:
            break

    print("\nFinal solution:")
    subset = [numbers_array[i] for i in range(n) if best_solution[i] == 1]
    print(f"Subset: {subset}")
    print(f"Sum: {sum(subset)}")
    print(f"Difference from target: {abs(sum(subset) - target)}")
    
    return best_solution, best_score

# ------------- Simulated Annealing algorithm ------------- 

# Functions for simulated_annealing algorithm
def temperature_function(scheme, T0, k, alpha=0.95, beta=1.0): # T0 is the initial temperature, k is the iteration number, alpha is the cooling rate, beta is the temperature reduction rate
    if scheme == 'geometric':
        return T0 * (alpha**k)
    if scheme == 'logarithmic':
        return T0 / math.log(1 + k + 1e-10) 
    if scheme == 'linear':
        return max(T0 - beta * k, 1e-8)

def get_neighbor_normal_dist(solution_vector, mu, sigma): # mu is the mean, sigma is the standard deviation
    n = len(solution_vector)
    neighbor = solution_vector.copy()
    i = int(abs(random.gauss(mu, sigma))) % n
    neighbor[i] = 1 - neighbor[i]
    return neighbor

def simulated_annealing_algorithm(numbers_array ,target, max_iterations = 1000, T_start = 100.0, temperature_scheme="logarithmic", alpha=0.95, beta=0.1):
    print('Simulated Annealing algorithm:')
    n = len(numbers_array)
    current_solution = generate_random_solution(n)
    current_score = objective_function(current_solution, numbers_array, target)
    best_solution = current_solution.copy()
    best_score = current_score
    mu = n / 2
    sigma = n / 4

    print(f'Numbers array = {numbers_array}')
    print(f'Target = {target}')
    print(f'Starting solution = {current_solution}')
    print(f'Starting score = {current_score}')

    for k in range(max_iterations):
        T = temperature_function(temperature_scheme, T_start, k, alpha, beta )
        neighbor = get_neighbor_normal_dist(current_solution, mu, sigma)
        neighbor_score = objective_function(neighbor, numbers_array, target)
        delta = neighbor_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_solution = neighbor
            current_score = neighbor_score

        if current_score < best_score:
            best_solution = current_solution.copy()
            best_score = current_score

        print(f"[{k}] T={T:.4f} Score={current_score}, Best={best_score}")

        if best_score == 0:
            print("Perfect solution found!")
            break

    subset = [numbers_array[i] for i in range(n) if best_solution[i] == 1]
    print("\nBest subset:", subset)
    print("Sum =", sum(subset), "Target =", target)
    print("Final difference:", abs(sum(subset) - target))
    
    return best_solution, best_score

# ------------- Genetic algorithms ------------- 

# Choosing Genetic Algorithm instead of Evolutionary Algorithm as it fits better SSP
def fitness(solution_vector, numbers, target_sum):
    score = objective_function(solution_vector, numbers, target_sum)
    return 1 / (1 + score)

# Selects parents based on each population object fitness score. pop_score / sum of fitness scores
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probabilities = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population=population,weights=selection_probabilities, k=1)[0]
    parent2 = random.choices(population=population,weights=selection_probabilities, k=1)[0]
    return parent1, parent2

# Crossover methods:
def one_point_crossover(parent1, parent2):
    n = len(parent1)
    if n <= 1:
        return parent1.copy(), parent2.copy()
    crossover_point = random.randint(1, n - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def uniform_crossover(parent1, parent2):
    n = len(parent1)
    child1, child2 = [], []
    for i in range(n):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

# Mutation methods:

# mutation % rate for each index
def bit_flip_mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


# Swaps 2 indexes with each other
def swap_mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        n = len(chromosome)
        if n > 1:
            idx1, idx2 = random.sample(range(n), 2) # random sample of 2 indexes from the chromosome
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome     

def genetic_algorithm(numbers_array ,target, max_generations = 1000, population_size = 10, elitism = True, crossover_method="one_point", mutation_method="bit_flip", mutation_rate=0.01, termination_condition="generations"):
    print('Genetic Algorithm')
    print(f'Numbers array = {numbers_array}')
    print(f'Target = {target}')
    print(f'Population size = {population_size}')
    print(f'Crossover method = {crossover_method}')
    print(f'Mutation method = {mutation_method}')
    print(f'Termination condition = {termination_condition}')
    
    n = len(numbers_array)

    population = [generate_random_solution(n) for _ in range(population_size)]
    
    # Initialize with the first solution to avoid None issues
    global_best_solution = population[0].copy()
    global_best_fitness = -1
    generations_without_improvement = 0
    last_best_fitness = -1

    for generation in range(max_generations):
        print(f'[{generation}] Generation')
        fitness_scores = [fitness(obj, numbers_array, target) for obj in population]        
        best_fitness = max(fitness_scores)
        best_index = fitness_scores.index(best_fitness)
        best_solution = population[best_index]

        if best_fitness > global_best_fitness:
            global_best_fitness = best_fitness
            global_best_solution = best_solution.copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Termination conditions
        if global_best_fitness == 1.0:
            print(f"Perfect solution found in generation {generation}!")
            break
        
        if termination_condition == "convergence" and generations_without_improvement >= 50:
            print(f"Convergence detected after {generation} generations without improvement!")
            break

        new_population = []

        if elitism:
            new_population.append(global_best_solution.copy())
        
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            
            if crossover_method == "one_point":
                child1, child2 = one_point_crossover(parent1, parent2)
            elif crossover_method == "uniform":
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                raise ValueError("Unknown crossover method")

            if mutation_method == "bit_flip":
                child1 = bit_flip_mutation(child1, mutation_rate)
                child2 = bit_flip_mutation(child2, mutation_rate)
            elif mutation_method == "swap":
                child1 = swap_mutation(child1, mutation_rate)
                child2 = swap_mutation(child2, mutation_rate)
            else:
                raise ValueError("Unknown mutation method")
            
            new_population.extend([child1,child2])
        
        population = new_population[:population_size]

    print("\nBest solution:", global_best_solution)
    subset = [numbers_array[i] for i in range(n) if global_best_solution[i] == 1]
    subset_sum = sum(subset)
    print("Subset:", subset)
    print("Subset sum:", subset_sum)
    print("Difference from target:", abs(target - subset_sum))
    return global_best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subset Sum Problem - Metaheuristics')
    parser.add_argument('--algorithm', choices=['brute_force', 'hill_climbing_det', 'hill_climbing_rand', 'tabu', 'simulated_annealing', 'genetic'], default='genetic', help='Algorithm to use')
    parser.add_argument('--csv_file', type=str, help='CSV file to load problem from (overrides random generation)')
    parser.add_argument('--size', type=int, default=10, help='Size of the numbers array')
    parser.add_argument('--min_val', type=int, default=0, help='Minimum value for random numbers')
    parser.add_argument('--max_val', type=int, default=50, help='Maximum value for random numbers')
    parser.add_argument('--target', type=int, help='Target sum (if not provided, will be random)')
    parser.add_argument('--iterations', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--population_size', type=int, default=20, help='Population size for GA')
    parser.add_argument('--crossover', choices=['one_point', 'uniform'], default='one_point', help='Crossover method for GA')
    parser.add_argument('--mutation', choices=['bit_flip', 'swap'], default='bit_flip', help='Mutation method for GA')
    parser.add_argument('--termination', choices=['generations', 'convergence'], default='generations', help='Termination condition for GA')
    
    args = parser.parse_args()
    
    # Load problem from CSV or generate random problem
    if args.csv_file:
        numbers, target = load_problem_from_csv(args.csv_file)
        print(f"Loaded problem from CSV file: {args.csv_file}")
    else:
        # Generate problem instance
        numbers = generate_random_number_vector(args.min_val, args.max_val, args.size)
        if args.target is None:
            target = random.randint(args.min_val, args.max_val)
        else:
            target = args.target
    
    print(f"Problem: Find subset of {numbers} that sums to {target}")
    print(f"Algorithm: {args.algorithm}")
    print("-" * 50)
    
    if args.algorithm == 'brute_force':
        brute_force_algorithm(numbers, target)
    elif args.algorithm == 'hill_climbing_det':
        hill_climbing_deterministic_algorithm(numbers, target, args.iterations)
    elif args.algorithm == 'hill_climbing_rand':
        hill_climbing_random_neighbor_algorithm(numbers, target, args.iterations)
    elif args.algorithm == 'tabu':
        tabu_search_algorithm(numbers, target, args.iterations, 10)
    elif args.algorithm == 'simulated_annealing':
        simulated_annealing_algorithm(numbers, target, args.iterations, 100, "geometric", 0.95, 0.1)
    elif args.algorithm == 'genetic':
        genetic_algorithm(numbers, target, args.iterations, args.population_size, True, 
                         args.crossover, args.mutation, 0.01, args.termination)
