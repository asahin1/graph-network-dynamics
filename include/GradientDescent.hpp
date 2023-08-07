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
    GradientDescent(std::shared_ptr<Graph> initGraph, bool weightConstraint);
    void runNStepDescent(const int nIter);
    std::vector<double> returnEigenvalues();
    void destroyHistogram();
    void destroyObjectivePlot();
private:
    // Attributes
    const int graphGridSize;
    const double minEdgeWeight{0.2};
    Gnuplot histogramStream;
    Gnuplot objectivePlotStream;
    double gradientStep{0.000001}; // works for 10x10
    double minGradNorm{0.6}; // works for 10x10
    std::vector<std::shared_ptr<Graph>> graphHistory;
    std::vector<double> objectiveValueHistory;
    const bool constrainedWeights;

    // Methods
    void decreaseGradientStep();
    void runOneStepDescent();
    Eigen::MatrixXf computeAdjGradientDoubleMin(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const;
    Eigen::MatrixXf computeAdjGradientDoubleSum(const std::shared_ptr<Graph> graph) const;
    Eigen::MatrixXf computeAdjGradientDoubleSumNew(const std::shared_ptr<Graph> graph) const;
    double evaluateObjectiveFunction(const Eigen::VectorXf& eigenvalues) const;
    Eigen::MatrixXf scaleGradientAroundWeightThreshold(const Eigen::MatrixXf &connectivityMatrix, const Eigen::MatrixXf &newAdjacencyMatrix, const double matrixSum) const;
    std::vector<MatrixIdx> getInvalidWeightIdx(const Eigen::MatrixXf &adjMat) const;
    void plotHistogram();
    void plotObjectivePlot();
    void plotGraph(bool waitForKey=true);
    void printIterInfo(const int iterNo) const;
};



#endif
