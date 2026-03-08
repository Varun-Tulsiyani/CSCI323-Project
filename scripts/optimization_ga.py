
import numpy as np
import pandas as pd

class GeneticOptimizer:
    """Genetic Algorithm optimizer for wave energy converter placement."""

    def __init__(self, model, pipeline, population_size=60, generations=120,
                 mutation_rate=0.2, mutation_sigma=15, bounds=(0,1000)):
        self.model = model
        self.pipeline = pipeline
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.bounds = bounds
        self.dim = 32

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def fitness(self, layout):
        df = pd.DataFrame([layout])
        X = self.pipeline.transform(df)
        power = self.model.predict(X)
        return np.sum(power)

    def tournament(self, population, fitness, k=3):
        idx = np.random.choice(len(population), k)
        return population[idx[np.argmax(fitness[idx])]]

    def crossover(self, p1, p2):
        mask = np.random.rand(self.dim) < 0.5
        return np.where(mask, p1, p2)

    def mutate(self, child):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                child[i] += np.random.normal(0, self.mutation_sigma)
        return np.clip(child, self.bounds[0], self.bounds[1])

    def run(self):
        population = self.initialize_population()
        history = []

        for g in range(self.generations):
            fitness_vals = np.array([self.fitness(ind) for ind in population])
            history.append(np.max(fitness_vals))

            new_population = [population[np.argmax(fitness_vals)]]

            while len(new_population) < self.population_size:
                p1 = self.tournament(population, fitness_vals)
                p2 = self.tournament(population, fitness_vals)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = np.array(new_population)
            print(f"Generation {g} | Best Power {np.max(fitness_vals):.2f}")

        best = population[np.argmax([self.fitness(i) for i in population])]
        return best, history
