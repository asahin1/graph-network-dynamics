#ifndef _NODE_H_
#define _NODE_H_

#include <unordered_map>
#include <vector>
#include <memory>

struct MatrixIdx{
    int j;
    int k;
    MatrixIdx(int jj, int kk): j{jj}, k{kk}{}
};

class Node
{
private:
public:
    const int x, y; // Coordinates (just for plotting)
    const int id;
    // std::vector<std::shared_ptr<Node>> neighbors;
    std::unordered_map<std::shared_ptr<Node>,std::shared_ptr<double>> neighbors;
    double z;     // State
    double z_old; // Previous state (for discrete time computations)
    double z_dot{0};
    double z_dot_old{0};

    double minEdgeWeight{0.2};
    double gradientStep{0.1};

    Node(int xx, int yy, int id);
    Node(int xx, int yy, int id, double zz);

    double getDist(const Node &n) const;
    double isNear(const Node &n) const;

    bool isNeighbor(const std::shared_ptr<Node> n) const;

    void runLocalGD();

    void print(const std::string head) const;
};

#endif
