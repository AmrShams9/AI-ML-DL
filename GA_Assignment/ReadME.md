# Genetic Algorithm Task Scheduler

This project is a Python-based **Genetic Algorithm Task Scheduler**. It optimizes task distribution across two CPU cores by balancing the workload and minimizing the completion time for tasks. 

The project uses genetic algorithm principles like **selection**, **crossover**, and **mutation** to find an optimal or near-optimal distribution of tasks, given the constraint that each core's load should not exceed a specified limit.

![Task Scheduling GIF](path/to/overview.gif) <!-- Add an animated GIF overview of your project here -->

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Components](#components)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Genetic Algorithm Optimization:** Uses a population-based algorithm to balance tasks between cores.
- **Multiple Configurations:** Supports various population sizes to experiment with different genetic algorithm parameters.
- **Fitness Evaluation:** Dynamically evaluates load distribution, aiming to minimize the maximum load on any core.
- **File-Based Input:** Reads task data from a file for streamlined testing and validation.

