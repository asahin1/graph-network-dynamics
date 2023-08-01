#ifndef _DISTRIBUTEDGRADIENTDESCENT_H_
#define _DISTRIBUTEDGRADIENTDESCENT_H_

#include "Graph.hpp"
#include "gnuplot-iostream.h"

#include <memory>
class DistributedGD{

public:
    DistributedGD(std::shared_ptr<Graph> initGraph);
    void runNStepDescent(const int nIter);
    void destroyHistogram();

private:
    std::shared_ptr<Graph> currentGraph;
    const int graphGridSize;
    Gnuplot histogramStream;
    void runOneStepDescent();
    void plotHistogram();
    void plotGraph();
    void printIterInfo(const int iterNo) const;

};

#endif
