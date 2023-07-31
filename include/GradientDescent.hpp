#ifndef _GRADIENTDESCENT_H_
#define _GRADIENTDESCENT_H_

#include <Eigen/Dense>
#include <memory>

#include "Graph.hpp"
#include "gnuplot-iostream.h"

struct MatrixIdx{
    int j;
    int k;
    MatrixIdx(int jj, int kk): j{jj}, k{kk}{}
};

class GradientDescent{

public:
    GradientDescent(std::shared_ptr<Graph> initGraph);
    GradientDescent(std::shared_ptr<Graph> initGraph, bool weightConstraint, bool weightSumConstraint);
    void runNStepDescent(const int nIter);
    void destroyHistogram();
private:
    // Attributes
    const int graphGridSize;
    const double minEdgeWeight{0.2};
    const int maxRecompute{5};
    Gnuplot histogramStream;
    double gradientStep{1};
    std::vector<std::shared_ptr<Graph>> graphHistory;
    const bool constrainedWeights;
    const bool fixedWeightSum;

    // Methods
    void decreaseGradientStep();
    void runOneStepDescent();
    Eigen::MatrixXf computeAdjGradientDoubleMin(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const;
    Eigen::MatrixXf computeAdjGradientDoubleSum(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const;
    bool invalidAdjacencyMatrix(const Eigen::MatrixXf &adjMat) const;
    std::vector<MatrixIdx> getInvalidWeightIdx(const Eigen::MatrixXf &adjMat) const;
    void plotHistogram();
    void plotGraph();
    void printIterInfo(const int iterNo) const;
};



#endif
