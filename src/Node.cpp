#include "../include/Node.h"
#include <algorithm>

Node::Node(NodeType t, std::string v) : type(t), value(v), left(nullptr), right(nullptr), fitness(0.0) {}

Node::~Node() 
{
    delete left;
    delete right;
}

double Node::evaluate(const Patient& p, ModelType model) {
    if (type == TERMINAL) 
    {
        if (value[0] == 'F') 
        {
            int featureIndex = std::stoi(value.substr(1));
            return p.features[featureIndex];
        }
        return std::stod(value);
    }

    if (type == FUNCTION) 
    {
        double leftVal = left ? left->evaluate(p, model) : 0;
        double rightVal = right ? right->evaluate(p, model) : 0;

        if (model == ARITHMETIC_CLASSIFIER) 
        {
            if (value == "+") return leftVal + rightVal;
            if (value == "-") return leftVal - rightVal;
            if (value == "*") return leftVal * rightVal;
            if (value == "/") return (rightVal == 0) ? 1.0 : leftVal / rightVal; 
        } 
        else if (model == DECISION_TREE) 
        {
            if (value == "IF_GT") return (leftVal > 0) ? rightVal : 0; 
        }
    }
    return 0;
}

Node* Node::clone() const 
{
    Node* newNode = new Node(type, value);
    if (left) newNode->left = left->clone();
    if (right) newNode->right = right->clone();
    newNode->fitness = this->fitness;
    return newNode;
}

int Node::getDepth() const 
{
    if (type == TERMINAL) return 1;
    int leftDepth = left ? left->getDepth() : 0;
    int rightDepth = right ? right->getDepth() : 0;
    return 1 + std::max(leftDepth, rightDepth);
}