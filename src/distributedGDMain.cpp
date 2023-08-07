#include "../include/DistributedGD.hpp"
#include "../include/Graph.hpp"

int main(int argc, char *argv[]){

    int gridSize{10};
    int maxIter{100};

    if(argc==2){
        gridSize = std::stoi(argv[1]);
    }
    else if(argc==3){
        gridSize = std::stoi(argv[1]);
        maxIter = std::stoi(argv[2]);
    }

    // Initialize graph
    std::shared_ptr<Graph> graphInit = std::make_shared<Graph>();
    graphInit->constructSimpleGraph(gridSize);

    // DistributedGD distGDObj(graphInit,weightConstraint,weightSumConstraint);
    DistributedGD distGDObj(graphInit);
    distGDObj.runNStepDescent(maxIter);
    distGDObj.destroyHistogram();

    return 0;
}
