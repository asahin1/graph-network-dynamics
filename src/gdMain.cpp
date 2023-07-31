#include "../include/GradientDescent.hpp"
#include "../include/Graph.hpp"

int main(int argc, char *argv[]){

    int gridSize{10};
    int nIterFirst{100};
    bool weightConstraint{true};
    bool weightSumConstraint{true};

    if(argc==2){
        gridSize = std::stoi(argv[1]);
    }
    else if(argc==3){
        gridSize = std::stoi(argv[1]);
        nIterFirst = std::stoi(argv[2]);
    }
    else if(argc==4){
        gridSize = std::stoi(argv[1]);
        nIterFirst = std::stoi(argv[2]);
        weightConstraint = std::stoi(argv[3]);
    }
    else if(argc>4){
        gridSize = std::stoi(argv[1]);
        nIterFirst = std::stoi(argv[2]);
        weightConstraint = std::stoi(argv[3]);
        weightSumConstraint = std::stoi(argv[4]);
    }

    // Initialize graph
    std::shared_ptr<Graph> graphInit = std::make_shared<Graph>();
    graphInit->constructSimpleGraph(gridSize);

    GradientDescent gdObj(graphInit,weightConstraint,weightSumConstraint);
    gdObj.runNStepDescent(nIterFirst);
    gdObj.destroyHistogram();

    return 0;
}
