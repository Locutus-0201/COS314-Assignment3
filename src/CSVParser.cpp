#include "../include/CSVParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

std::vector<Patient> CSVParser::parseData(const std::string& filepath) {
    std::vector<Patient> dataset;
    std::ifstream file(filepath);
    std::string line;

    if (!file.is_open()) 
    {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return dataset;
    }

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) 
    {
        //replace all misc spaces and stuff
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');

        std::stringstream ss(line);
        Patient p;
        
        //straight to data
        if (ss >> p.label) 
        { 
            double featureVal;
            // Read all 9 remaining numbers on the line into the features vector
            while (ss >> featureVal) 
            {
                p.features.push_back(featureVal);
            }
            // Only add the patient if we successfully read the label and features
            dataset.push_back(p); 
        }
    }
    return dataset;
}