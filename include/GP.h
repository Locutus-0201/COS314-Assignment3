#ifndef GP_H
#define GP_H

#include "Common.h"
#include "Node.h"
#include <vector>
#include <random>

// Struct to hold results for the report
struct Metrics 
{
    double accuracy;
    double f_measure;
};

struct RunResult 
{
    double trainAcc;
    double testAcc;
    double fMeasure;
    double runtime; // in seconds
};

class GP 
{
private:
    void localSearch(Node* tree, const std::vector<Patient>& data);
    std::vector<Node*> population;
    ModelType model;
    std::mt19937 rng;

    Node* generateTree(int currentDepth, int maxDepth, bool isFull);
    Metrics calculateMetrics(Node* tree, const std::vector<Patient>& data);
    void evaluatePopulationConcurrent(const std::vector<Patient>& data);
    Node* tournamentSelection();
    Node* crossover(Node* parent1, Node* parent2);
    void mutate(Node* tree);
    void clearPopulation();

    // Helpers for true subtree crossover
    void collectNodes(Node* root, std::vector<Node*>& nodes);

public:
    GP(ModelType m, long seed);
    ~GP();

    Node* train(const std::vector<Patient>& trainData);
    Metrics test(Node* bestModel, const std::vector<Patient>& testData);
};

#endif