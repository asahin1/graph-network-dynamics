#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "Node.hpp"
#include <Eigen/Dense>

class Graph
{
private:
public:
    int gridSize;
    std::vector<std::shared_ptr<Node>> nodes;
    Eigen::MatrixXf adjacencyMatrix, degreeMatrix, laplacianMatrix;
    Eigen::MatrixXf connectivityMatrix;
    Eigen::VectorXf eigenValues;
    Eigen::MatrixXf eigenVectors;

    static constexpr double eps{0.1};

    void constructSimpleGraph(const int size);

    std::shared_ptr<Graph> applyGradient(const Eigen::MatrixXf &newAdjacencyMatrix) const;

    void computeMatrices();
    void eigenDecompose();

    void simulateDynamics(const int tMax);
};

#endif
