#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>

struct Patient 
{
    int label; 
    std::vector<double> features; 
};

enum ModelType { ARITHMETIC_CLASSIFIER, DECISION_TREE };

//Fixed
const int POPULATION_SIZE = 200;
const int MAX_GENERATIONS = 100;

// dd(design desicion)
const int MAX_DEPTH = 5;            // Keeps equations relatively small to prevent overfitting
const int TOURNAMENT_SIZE = 3;      // Lower selection pressure to maintain diversity
const double CROSSOVER_RATE = 0.85; // 85% of population breeds
const double MUTATION_RATE = 0.15;  // 15% mutation rate for exploration

#endif