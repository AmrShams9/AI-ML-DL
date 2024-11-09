import random

# evaluate chromosome fnc
def evaluate_chromosome(chromosome, task_times, max_time_limit):
    core1_time = sum(task_times[i] for i in range(len(chromosome)) if chromosome[i] == 1)
    core2_time = sum(task_times[i] for i in range(len(chromosome)) if chromosome[i] == 0)
    
    if core1_time > max_time_limit or core2_time > max_time_limit:
        return float('inf')  

    return max(core1_time, core2_time)

# make random population fnc
def generate_population(size, num_tasks):
    return [ [random.randint(0, 1) for _ in range(num_tasks)] for _ in range(size) ]

# Roulette wheel selection fnc
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(1/f for f in fitness_values if f != float('inf'))
    pick = random.uniform(0, total_fitness)
    current = 0

    for i, f in enumerate(fitness_values):
        if f != float('inf'):
            current += 1/f
            if current > pick:
                return population[i]

# One-point crossover fnc
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Flip bit mutation fnc
def flip_bit_mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# genetic algorithm fnc
def genetic_algorithm(task_times, max_time_limit, population_size=100, generations=1000, mutation_rate=0.01):
    num_tasks = len(task_times)
    population = generate_population(population_size, num_tasks)

    for generation in range(generations):
        fitness_values = [evaluate_chromosome(chrom, task_times, max_time_limit) for chrom in population]
        new_population = []

        # Keep the best sol
        best_index = fitness_values.index(min(fitness_values))
        new_population.append(population[best_index])

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = flip_bit_mutation(child1, mutation_rate)
            child2 = flip_bit_mutation(child2, mutation_rate)
            
            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final evaluation
    best_chromosome = min(population, key=lambda chrom: evaluate_chromosome(chrom, task_times, max_time_limit))
    best_score = evaluate_chromosome(best_chromosome, task_times, max_time_limit)

    core1_tasks = [task_times[i] for i in range(num_tasks) if best_chromosome[i] == 1]
    core2_tasks = [task_times[i] for i in range(num_tasks) if best_chromosome[i] == 0]

    return best_score, best_chromosome, core1_tasks, core2_tasks

def read_input(file_path):
    with open(file_path, 'r') as file:
        num_cases = int(file.readline().strip())
        cases = []
        
        for _ in range(num_cases):
            max_time_limit = int(file.readline().strip())
            num_tasks = int(file.readline().strip())
            task_times = [int(file.readline().strip()) for _ in range(num_tasks)]
            cases.append((max_time_limit, task_times))
        
    return cases

def main():
    input_file = 'D:\Yasse mourad\GA_Assignment\input file for genetic algo.txt'
    cases = read_input(input_file)

    # Try different population sizes
    population_sizes = [50, 100, 250]

    for i, (max_time_limit, task_times) in enumerate(cases):
        print(f"Test case {i + 1}:")

        for pop_size in population_sizes:
            print(f"\nPopulation size: {pop_size}")
            best_score, best_chromosome, core1_tasks, core2_tasks = genetic_algorithm(
                task_times, max_time_limit, population_size=pop_size
            )
            print(f"Best solution evaluation score: {best_score}")
            print(f"Chromosome representation: {best_chromosome}")
            print(f"Core 1 tasks: {core1_tasks} (Total time: {sum(core1_tasks)})")
            print(f"Core 2 tasks: {core2_tasks} (Total time: {sum(core2_tasks)})")
        print("=" * 40)

if __name__ == "__main__":
    main()
