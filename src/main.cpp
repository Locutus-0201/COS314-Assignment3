#include "../include/CSVParser.h"
#include "../include/GP.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Helper to calculate Independent Two-Sample T-Test
void calculateTTest(const vector<RunResult>& arithResults, const vector<RunResult>& dtResults) 
{
    double sumA = 0, sumD = 0;
    for (int i = 0; i < 30; i++) 
    {
        sumA += arithResults[i].testAcc;
        sumD += dtResults[i].testAcc;
    }
    double meanA = sumA / 30.0;
    double meanD = sumD / 30.0;

    double varA = 0, varD = 0;
    for (int i = 0; i < 30; i++) 
    {
        varA += pow(arithResults[i].testAcc - meanA, 2);
        varD += pow(dtResults[i].testAcc - meanD, 2);
    }
    varA /= 29.0; 
    varD /= 29.0;

    double tStat = (meanA - meanD) / sqrt((varA / 30.0) + (varD / 30.0));
    
    cout << "\n#### Statistical Significance (T-Test) ####\n";
    cout << "Arithmetic Mean Test Acc: " << meanA << "%\n";
    cout << "Decision Tree Mean Test Acc: " << meanD << "%\n";
    cout << "T-Statistic: " << abs(tStat) << "\n";
    if (abs(tStat) > 2.00) cout << "Result: SIGNIFICANT difference in performance (p < 0.05).\n";
    else cout << "Result: NO significant difference in performance.\n";
}

int main() {
    long baseSeed;
    cout << "Enter random seed value: ";
    cin >> baseSeed;

    // --- AUTO-PRINT TABLE 1 (PARAMETERS) ---
    cout << "\n\n";
    cout << "Design Decisions: Genetic Programming Parameters\n";
    cout << "\n";
    cout << left << setw(30) << "Parameter" << "Value\n";
    cout << "--------------------------------------------------------\n";
    cout << left << setw(30) << "Population size" << POPULATION_SIZE << "\n";
    cout << left << setw(30) << "Initial tree generation" << "ramped half-and-half\n";
    cout << left << setw(30) << "Initial tree depth (dd)" << "2 to " << MAX_DEPTH << "\n";
    cout << left << setw(30) << "Max offspring depth (dd)" << MAX_DEPTH << "\n";
    cout << left << setw(30) << "Selection method (dd)" << "Tournament\n";
    cout << left << setw(30) << "Tournament size (dd)" << TOURNAMENT_SIZE << "\n";
    cout << left << setw(30) << "Function set (dd)" << "+, -, *, / (Arith) | IF_GT (DT)\n";
    cout << left << setw(30) << "Crossover rate (dd%)" << (CROSSOVER_RATE * 100) << "%\n";
    cout << left << setw(30) << "Mutation rate (dd%)" << (MUTATION_RATE * 100) << "%\n";
    cout << left << setw(30) << "Mutation type" << "point / subtree\n";
    cout << left << setw(30) << "Fitness function" << "accuracy (w/ parsimony)\n";
    cout << left << setw(30) << "Maximum generations" << MAX_GENERATIONS << "\n";
    cout << "--------------------------------------------------------\n\n";

    vector<Patient> trainData = CSVParser::parseData("data/Breast_train.csv");
    vector<Patient> testData = CSVParser::parseData("data/Breast_test.csv");

    if (trainData.empty() || testData.empty()) {
        cout << "Data load failed.\n";
        return -1;
    }

    vector<RunResult> arithResults(30);
    vector<RunResult> dtResults(30);
    RunResult bestArith = {0,0,0,0}, bestDT = {0,0,0,0};

    cout << "Starting 30 Independent Runs for ARITHMETIC CLASSIFIER...\n";
    for(int i = 0; i < 30; i++) {
        auto start = high_resolution_clock::now();
        
        GP model(ARITHMETIC_CLASSIFIER, baseSeed + i);
        Node* bestModel = model.train(trainData);
        Metrics testM = model.test(bestModel, testData);
        Metrics trainM = model.test(bestModel, trainData);
        
        auto stop = high_resolution_clock::now();
        double rTime = duration_cast<milliseconds>(stop - start).count() / 1000.0;
        
        arithResults[i] = {trainM.accuracy, testM.accuracy, testM.f_measure, rTime};
        if (testM.accuracy > bestArith.testAcc) bestArith = arithResults[i];
        
        delete bestModel;
        cout << "Run " << i+1 << "/30 Completed.\r" << flush;
    }

    cout << "\nStarting 30 Independent Runs for DECISION TREE...\n";
    for(int i = 0; i < 30; i++) {
        auto start = high_resolution_clock::now();
        
        GP model(DECISION_TREE, baseSeed + i);
        Node* bestModel = model.train(trainData);
        Metrics testM = model.test(bestModel, testData);
        Metrics trainM = model.test(bestModel, trainData);
        
        auto stop = high_resolution_clock::now();
        double rTime = duration_cast<milliseconds>(stop - start).count() / 1000.0;
        
        dtResults[i] = {trainM.accuracy, testM.accuracy, testM.f_measure, rTime};
        if (testM.accuracy > bestDT.testAcc) bestDT = dtResults[i];
        
        delete bestModel;
        cout << "Run " << i+1 << "/30 Completed.\r" << flush;
    }

    // PRINT FINAL REPORT TABLE
    cout << "\n\n";
    cout << "Tabulated Results: Comparison of Classification Performance (BEST RUNS)\n";
    cout << "\n";
    cout << left << setw(20) << "Algorithm" << setw(15) << "Training (%)" << setw(15) << "Test (%)" << setw(15) << "F-measure" << setw(15) << "Runtime (s)" << "\n";
    cout << "--------------------------------------------------------\n";
    cout << left << setw(20) << "Decision Tree" << setw(15) << bestDT.trainAcc << setw(15) << bestDT.testAcc << setw(15) << bestDT.fMeasure << setw(15) << bestDT.runtime << "\n";
    cout << left << setw(20) << "GP Classifier" << setw(15) << bestArith.trainAcc << setw(15) << bestArith.testAcc << setw(15) << bestArith.fMeasure << setw(15) << bestArith.runtime << "\n";
    
    calculateTTest(arithResults, dtResults);

    return 0;
}