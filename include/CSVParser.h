#ifndef CSVPARSER_H
#define CSVPARSER_H

#include "Common.h"
#include <vector>
#include <string>

class CSVParser 
{
public:
    static std::vector<Patient> parseData(const std::string& filepath);
};

#endif