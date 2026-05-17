#include "../include/GP.h"
#include <iostream>
#include <thread>
#include <algorithm>

using namespace std;

GP::GP(ModelType m, long seed) : model(m) { rng.seed(seed); }
GP::~GP() { clearPopulation(); }

void GP::clearPopulation() 
{
    for (Node* tree : population) delete tree;
    population.clear();
}

Node* GP::generateTree(int currentDepth, int maxDepth, bool isFull) {
    if (currentDepth >= maxDepth || (!isFull && currentDepth > 0 && rng() % 100 < 20)) {
        return new Node(TERMINAL, "F" + to_string(rng() % 9));
    }
    string funcName;
    if (model == ARITHMETIC_CLASSIFIER) 
    {
        string ops[] = {"+", "-", "*", "/"};
        funcName = ops[rng() % 4];
    } else 
    {
        funcName = "IF_GT"; // Placeholder for Decision Tree logic
    }
    Node* node = new Node(FUNCTION, funcName);
    node->left = generateTree(currentDepth + 1, maxDepth, isFull);
    node->right = generateTree(currentDepth + 1, maxDepth, isFull);
    return node;
}

// Calculates Accuracy AND F-Measure
Metrics GP::calculateMetrics(Node* tree, const vector<Patient>& data) {
    double TP = 0, FP = 0, TN = 0, FN = 0;
    
    for (const auto& p : data) 
    {
        double result = tree->evaluate(p, model);
        int prediction = (result > 0) ? 1 : 0; 
        
        if (prediction == 1 && p.label == 1) TP++;
        else if (prediction == 1 && p.label == 0) FP++;
        else if (prediction == 0 && p.label == 0) TN++;
        else if (prediction == 0 && p.label == 1) FN++;
    }
    
    Metrics m;
    m.accuracy = ((TP + TN) / data.size()) * 100.0;
    
    double precision = (TP + FP == 0) ? 0 : TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : TP / (TP + FN);
    m.f_measure = (precision + recall == 0) ? 0 : 2 * ((precision * recall) / (precision + recall));
    
    return m;
}

void GP::evaluatePopulationConcurrent(const vector<Patient>& data) 
{
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; 
    vector<thread> threads;
    int chunkSize = population.size() / numThreads;

    auto worker = [&](int start, int end) 
    {
        for (int i = start; i < end; ++i) 
        {
            Metrics m = calculateMetrics(population[i], data);
            // Parsimony Penalty: Prevent overfitting by punishing huge trees
            population[i]->fitness = m.accuracy - (population[i]->getDepth() * 0.1);
        }
    };

    for (int i = 0; i < numThreads; ++i) 
    {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? population.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }
    for (auto& t : threads) t.join();
}

Node* GP::tournamentSelection() 
{
    Node* best = population[rng() % POPULATION_SIZE];
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) 
    {
        Node* competitor = population[rng() % POPULATION_SIZE];
        if (competitor->fitness > best->fitness) best = competitor;
    }
    return best;
}

// True Subtree Crossover Fix
void GP::collectNodes(Node* root, vector<Node*>& nodes) 
{
    if (!root) return;
    nodes.push_back(root);
    collectNodes(root->left, nodes);
    collectNodes(root->right, nodes);
}

Node* GP::crossover(Node* parent1, Node* parent2) 
{
    Node* child = parent1->clone();
    vector<Node*> childNodes, p2Nodes;
    collectNodes(child, childNodes);
    collectNodes(parent2, p2Nodes);
    
    if (childNodes.size() > 1 && p2Nodes.size() > 1) 
    {
        Node* targetNode = childNodes[1 + (rng() % (childNodes.size() - 1))];
        Node* sourceNode = p2Nodes[1 + (rng() % (p2Nodes.size() - 1))];
        
        delete targetNode->left;
        delete targetNode->right;
        targetNode->type = sourceNode->type;
        targetNode->value = sourceNode->value;
        targetNode->left = sourceNode->left ? sourceNode->left->clone() : nullptr;
        targetNode->right = sourceNode->right ? sourceNode->right->clone() : nullptr;
    }
    return child;
}

void GP::mutate(Node* tree) 
{
    vector<Node*> nodes;
    collectNodes(tree, nodes);
    if (nodes.empty()) return;
    Node* target = nodes[rng() % nodes.size()];
    
    if (target->type == TERMINAL) 
    {
        target->value = "F" + to_string(rng() % 9);
    } else if (model == ARITHMETIC_CLASSIFIER) 
    {
        string ops[] = {"+", "-", "*", "/"};
        target->value = ops[rng() % 4];
    }
}

Node* GP::train(const vector<Patient>& trainData) 
{
    clearPopulation();
    for (int i = 0; i < POPULATION_SIZE; ++i) 
    {
        population.push_back(generateTree(0, 2 + (i % (MAX_DEPTH - 1)), i < POPULATION_SIZE / 2));
    }
    Node* globalBest = nullptr;

    for (int gen = 0; gen < MAX_GENERATIONS; ++gen) 
    {
        evaluatePopulationConcurrent(trainData); 
        Node* genBest = population[0];
        for (Node* n : population) 
        {
            if (n->fitness > genBest->fitness) genBest = n;
        }
        if (!globalBest || genBest->fitness > globalBest->fitness) 
        {
            if (globalBest) delete globalBest;
            globalBest = genBest->clone();
        }

        vector<Node*> newPopulation;
        newPopulation.push_back(globalBest->clone()); 
        while (newPopulation.size() < POPULATION_SIZE) 
        {
            Node* child = crossover(tournamentSelection(), tournamentSelection());
            if (rng() % 100 < MUTATION_RATE * 100) mutate(child);
            newPopulation.push_back(child);
        }
        clearPopulation();
        population = newPopulation;
    }
    return globalBest;
}

Metrics GP::test(Node* bestModel, const vector<Patient>& testData) 
{
    return calculateMetrics(bestModel, testData);
}