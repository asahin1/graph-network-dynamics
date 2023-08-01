#include "Node.hpp"

#include <math.h>
#include <iostream>
#include <memory>

Node::Node(int xx, int yy, int id) : x(xx), y(yy), id(id), z(0), z_old(0) {}
Node::Node(int xx, int yy, int id, double zz) : x(xx), y(yy), id(id), z(zz), z_old(zz) {}

double Node::getDist(const Node &n) const
{
    double dx = x - n.x;
    double dy = y - n.y;
    return sqrt(dx * dx + dy * dy);
}
double Node::isNear(const Node &n) const
{
    return getDist(n) < sqrt(2);
}

bool Node::isNeighbor(const std::shared_ptr<Node> n) const
{
    //     for (int i{0}; i < neighbors.size(); i++)
    //     {
    //         if (neighbors[i] == n)
    //             return true;
    //     }
    if(neighbors.find(n) != neighbors.end())
        return true;
    return false;
}

void Node::runLocalGD(){
    // compute and update own weight info
    for(auto &n: neighbors){
        if(n.first->x % 2 ==0)
            n.second+=0.01;
        else
            n.second-=0.01;
    }
    // communicate the updated weights to neighbors
}

void Node::print(const std::string head) const
{
    std::cout << head << " : "
              << "x = " << x << ", y = " << y << std::endl;
}
