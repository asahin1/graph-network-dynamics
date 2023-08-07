#include "DistributedGD.hpp"
#include "Histogram.hpp"
#include "Plot.hpp"
#include <memory>
#include <numeric>

DistributedGD::DistributedGD(std::shared_ptr<Graph> initGraph) : currentGraph{initGraph}, graphGridSize{currentGraph->gridSize}{}

void DistributedGD::runNStepDescent(const int nIter){
    
    plotHistogram();
    plotGraph();
    for(int i{0}; i<nIter; i++){
        printIterInfo(i);
        runOneStepDescent();
        // Gradient descent stuck check?
        plotHistogram();
    }
    plotGraph();
    std::cout << currentGraph->adjacencyMatrix << std::endl;
}

void DistributedGD::runOneStepDescent(){

    for(auto &n: currentGraph->nodes){
        n->runLocalGD();
    }
    // regenerate or update the graph usign new node info
    currentGraph->computeMatrices();
}

double DistributedGD::evaluateObjectiveFunction(const Eigen::VectorXf& eigenvalues) const{
    double objVal{0};
    double h{0};
    double denominator{0};
    double a{0.1};
    for(int i{0}; i<eigenvalues.size(); i++){
        double lambda_i = eigenvalues[i];
        h = a*sqrt(lambda_i);
        for(int j{0}; j<eigenvalues.size(); j++){
            double lambda_j = eigenvalues[j];
            denominator = pow(sqrt(lambda_i)-sqrt(lambda_j),2) + h*h;
            objVal += (4*h*h/lambda_j)*(1/(denominator*denominator));
        }
    }
    return objVal;
}

void DistributedGD::plotHistogram(){

    if(currentGraph)
        histogram::generateHistogram(histogramStream,currentGraph->eigenValues);

}

void DistributedGD::destroyHistogram(){
    histogramStream << "set term x11 close\n";
}

void DistributedGD::plotGraph(){

    Plot graphPlotter("Graph Plot", 500/graphGridSize, 2, 2, graphGridSize, graphGridSize);
    graphPlotter.plotGraph(*currentGraph);
    graphPlotter.displayPlot(true);

}

void DistributedGD::printIterInfo(const int iterNo) const{
    std::cout << "Iter #: " << iterNo+1 << std::endl;
    auto eigenvalues = currentGraph->eigenValues;
    std::cout << "Eigenvalue sum: " << std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0) << std::endl;
    double eigDistNorm{0};
    double eigDistMin{eigenvalues[eigenvalues.size()-1]};
    for(int j{0}; j<eigenvalues.size(); j++){
        for(int k{0}; k<eigenvalues.size(); k++){
            eigDistNorm += pow(eigenvalues[j] - eigenvalues[k], 2);
            if(abs(eigenvalues[j]-eigenvalues[k])<eigDistMin)
                eigDistMin = abs(eigenvalues[j]-eigenvalues[k]);
        }
    }
    std::cout << "Eigenvalues cumulative distance : " << eigDistNorm << std::endl;
    std::cout << "Eigenvalues minimum distance : " << eigDistMin << std::endl;
    std::cout << "Objective value: " << evaluateObjectiveFunction(eigenvalues) << std::endl;
}
