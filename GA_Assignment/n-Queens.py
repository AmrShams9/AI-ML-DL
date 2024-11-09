import random

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def fitness(self):
        """Calculates the fitness of an individual based on the number of non-attacking pairs of queens."""
        fitness = 0
        for i in range(len(self.chromosome)):
            for j in range(i + 1, len(self.chromosome)):
                if abs(self.chromosome[i] - self.chromosome[j]) == abs(i - j):
                    fitness += 1
        return fitness

class Population:
    def __init__(self, size):
        self.individuals = [Individual([random.randint(0, 7) for _ in range(8)]) for _ in range(size)]

    def select_parents(self, fitness_scores):
        """Selects two parents using tournament selection."""
        tournament_size = 5
        tournament_participants = random.sample(self.individuals, tournament_size)
        tournament_fitness_scores = [individual.fitness() for individual in tournament_participants]
        best_individual_index = tournament_fitness_scores.index(min(tournament_fitness_scores))
        best_individual = tournament_participants[best_individual_index]

        tournament_participants = random.sample(self.individuals, tournament_size)
        tournament_fitness_scores = [individual.fitness() for individual in tournament_participants]
        second_best_individual_index = tournament_fitness_scores.index(min(tournament_fitness_scores))
        second_best_individual = tournament_participants[second_best_individual_index]

        return best_individual, second_best_individual

    def crossover(self, parent1, parent2):
        """Performs single-point crossover."""
        crossover_point = random.randint(1, 7)
        child1 = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2 = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        return Individual(child1), Individual(child2)

    def mutate(self, individual, mutation_rate):
        """Performs mutation with a given probability."""
        for i in range(len(individual.chromosome)):
            if random.random() < mutation_rate:
                individual.chromosome[i] = random.randint(0, 7)
        return individual

def genetic_algorithm(population_size, generations, mutation_rate):
    population = Population(population_size)

    for generation in range(generations):
        fitness_scores = [individual.fitness() for individual in population.individuals]
        new_population = []

        # Keep the best individual
        best_individual = min(population.individuals, key=lambda x: x.fitness())
        new_population.append(best_individual)

        while len(new_population) < population_size:
            parent1, parent2 = population.select_parents(fitness_scores)
            child1, child2 = population.crossover(parent1, parent2)
            child1 = population.mutate(child1, mutation_rate)
            child2 = population.mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population.individuals = new_population

    best_individual = min(population.individuals, key=lambda x: x.fitness())
    if best_individual.fitness() == 0:
        return best_individual.chromosome
    else:
        return None

# Example usage:
solution = genetic_algorithm(100, 1000, 0.01)
if solution:
    print("Solution found:")
    for i, queen_row in enumerate(solution):
        print(f"Queen {i+1} on row {queen_row+1}")
else:
    print("No solution found.")