#ifndef NODE_H
#define NODE_H

#include "Common.h"
#include <string>

enum NodeType { FUNCTION, TERMINAL };

class Node 
{
public:
    NodeType type;
    std::string value;
    Node* left;
    Node* right;
    double fitness; // Added to store the tree's accuracy

    Node(NodeType t, std::string v);
    ~Node();

    double evaluate(const Patient& p, ModelType model);
    Node* clone() const;
    int getDepth() const;
};

#endif