import random
import math

# Floating Point GA
class GeneticAlgorithm:
    def __init__(self, input_file, output_file, population_size=50, generations=100, mutation_rate=0.1, tournament_size=3):
        self.input_file = input_file
        self.output_file = output_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size  # Tournament size

    def parse_input(self):
        """Parse the input file and validate its format."""
        with open(self.input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # Remove blank lines and extra spaces

        if not lines:
            raise ValueError("Input file is empty.")

        # Validate the first line (number of datasets)
        try:
            num_datasets = int(lines[0])
        except ValueError:
            raise ValueError(f"Invalid number of datasets in the first line: '{lines[0]}'. Must be an integer.")
        
        if num_datasets < 1:
            raise ValueError("Number of datasets must be at least 1.")

        datasets = []
        idx = 1
        for dataset_index in range(num_datasets):
            if idx >= len(lines):
                raise ValueError(f"Dataset {dataset_index + 1}: Missing data. Check input format.")

            # Read number of chemicals and total proportion constraint
            try:
                num_chemicals, total_proportion = map(float, lines[idx].split())
                num_chemicals = int(num_chemicals)
            except ValueError:
                raise ValueError(f"Dataset {dataset_index + 1}: Invalid number of chemicals or constraint.")
            idx += 1

            if idx >= len(lines):
                raise ValueError(f"Dataset {dataset_index + 1}: Missing bounds line.")
            try:
                bounds_data = list(map(float, lines[idx].split()))
                print(f"Bounds data for Dataset {dataset_index + 1}: {bounds_data}")  # Print raw bounds data
                if len(bounds_data) % 2 != 0:
                    raise ValueError(f"Dataset {dataset_index + 1}: Bounds must be in pairs (lower, upper).")
                bounds = [(bounds_data[i], bounds_data[i + 1]) for i in range(0, len(bounds_data), 2)]
                print(f"Parsed bounds for Dataset {dataset_index + 1}: {bounds}")  # Print parsed bounds
                if len(bounds) != num_chemicals:
                    raise ValueError(f"Dataset {dataset_index + 1}: Mismatch in number of bounds and chemicals. "
                                     f"Expected {num_chemicals} pairs, but got {len(bounds)} pairs.")
            except ValueError as ve:
                raise ValueError(f"Dataset {dataset_index + 1}: Invalid bounds format. Error: {ve}")
            idx += 1

            if idx >= len(lines):
                raise ValueError(f"Dataset {dataset_index + 1}: Missing cost coefficients.")
            try:
                costs = list(map(float, lines[idx].split()))
            except ValueError:
                raise ValueError(f"Dataset {dataset_index + 1}: Invalid cost coefficients.")
            idx += 1

            if len(costs) != num_chemicals:
                raise ValueError(f"Dataset {dataset_index + 1}: Mismatch in number of cost coefficients and chemicals. "
                                 f"Expected {num_chemicals}, but got {len(costs)}.")

            datasets.append((num_chemicals, total_proportion, bounds, costs))

        return datasets

    def fitness(self, chromosome, costs, total_proportion):
        cost = sum(p * c for p, c in zip(chromosome, costs))
        penalty = 10 * abs(total_proportion - sum(chromosome))
        return cost + penalty

    def normalize(self, chromosome, bounds, total_proportion):
        """Normalize a solution to satisfy the total proportion constraint."""
        scale = total_proportion / sum(chromosome)
        for i in range(len(chromosome)):
            chromosome[i] = max(bounds[i][0], min(bounds[i][1], chromosome[i] * scale))

    def generate_population(self, bounds, total_proportion):
        """Generate an initial population of solutions."""
        population = []
        for _ in range(self.population_size):
            proportions = [random.uniform(lb, ub) for lb, ub in bounds]
            self.normalize(proportions, bounds, total_proportion)
            population.append(proportions)
        return population

    def two_point_crossover(self, parent1, parent2, Pc=0.7):
        if random.random() <= Pc:
            L = len(parent1)
            point1, point2 = sorted(random.sample(range(1, L), 2))
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
        else:
            return parent1[:], parent2[:]

    def non_uniform_mutation(self, chromosome, bounds, current_gen, max_gen, b=1.5):
        for i in range(len(chromosome)):
            if random.random() <= self.mutation_rate:
                r_i1 = random.random()
                delta = (chromosome[i] - bounds[i][0]) if r_i1 <= 0.5 else (bounds[i][1] - chromosome[i])
                delta *= (1 - random.random() * ((1 - current_gen / max_gen) * b))
                chromosome[i] = max(bounds[i][0], min(bounds[i][1], chromosome[i] + delta))

    def tournament_selection(self, population, costs, total_proportion, tournament_size=3):
        """Select two parents using tournament selection."""
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda sol: self.fitness(sol, costs, total_proportion))
        return tournament[0], tournament[1]  # Return the two best individuals from the tournament

    def genetic_algorithm(self, dataset):
        """Optimize chemical proportions using a genetic algorithm."""
        num_chemicals, total_proportion, bounds, costs = dataset
        population = self.generate_population(bounds, total_proportion)
        
        for generation in range(self.generations):
            population.sort(key=lambda sol: self.fitness(sol, costs, total_proportion))
            next_population = population[:10]  # Keep the top 10 solutions (elitist replacement)
            
            while len(next_population) < self.population_size:
                parent1, parent2 = self.tournament_selection(population[:20], costs, total_proportion, self.tournament_size)
                child1, child2 = self.two_point_crossover(parent1, parent2)
                self.non_uniform_mutation(child1, bounds, generation, self.generations)
                self.non_uniform_mutation(child2, bounds, generation, self.generations)
                self.normalize(child1, bounds, total_proportion)
                self.normalize(child2, bounds, total_proportion)
                next_population.extend([child1, child2])
            
            population = next_population
        
        best_solution = min(population, key=lambda sol: self.fitness(sol, costs, total_proportion))
        print(f"Chromosome Representation: {' '.join(map(lambda x: f'{x:.2f}', best_solution))}")  # Print chromosome
        return best_solution, self.fitness(best_solution, costs, total_proportion)

    def run(self):
        """Run the genetic algorithm and write results to a file."""
        datasets = self.parse_input()
        with open(self.output_file, 'w') as f:
            for i, dataset in enumerate(datasets, start=1):
                chromosome, total_cost = self.genetic_algorithm(dataset)
                f.write(f"Dataset {i}\n")
                f.write(f"Chemical Proportions: {' '.join(map(lambda x: f'{x:.2f}', chromosome))}\n")
                f.write(f"Total Cost: {total_cost:.2f}\n\n")
                print(f"Dataset {i} - Total Cost: {total_cost:.2f}")

if __name__ == "__main__":
    input_file = "GA_Assignment_2\input.txt"
    output_file = "output.txt"
    ga = GeneticAlgorithm(input_file, output_file)
    try:
        ga.run()
    except Exception as e:
        print(f"Error: {e}")
