#include "../include/GradientDescent.hpp"
#include "../include/Graph.hpp"

// My headers
#include "include/Plot.hpp"
#include "include/Dynamics.hpp"
#include "include/Force.hpp"

int findModeIndex(Eigen::VectorXf nums) {
    std::unordered_map<float, float> freq;
    int maxFreq = 0;
    int modeIndex = -1;

    // Count the frequency of each element
    for (int i = 0; i < nums.size(); i++) {
        freq[nums[i]]++;
        if (freq[nums[i]] > maxFreq) {
            maxFreq = freq[nums[i]];
            modeIndex = i;
        }
    }
    std::cout << "Mode FreqCount: " << maxFreq;
    if (modeIndex != -1) {
        std::cout << "\nMode Index: " << modeIndex << " Mode Value: " << nums[modeIndex] << std::endl;
    } else {
        std::cout << "\nNo mode found." << std::endl;
    }
    return modeIndex;
}

int findModeIndex(std::vector<double> nums) {
    std::unordered_map<float, float> freq;
    int maxFreq = 0;
    int modeIndex = -1;

    // Count the frequency of each element
    for (int i = 0; i < nums.size(); i++) {
        freq[nums[i]]++;
        if (freq[nums[i]] > maxFreq) {
            maxFreq = freq[nums[i]];
            modeIndex = i;
        }
    }
    std::cout << "Mode FreqCount: " << maxFreq;
    if (modeIndex != -1) {
        std::cout << "\nMode Index: " << modeIndex << " Mode Value: " << nums[modeIndex] << std::endl;
    } else {
        std::cout << "\nNo mode found." << std::endl;
    }
    return modeIndex;
}

void printEigenVal(std::vector<double> ev){
    std::cout << "\n\n";
    for(int i{0}; i < ev.size(); i++){
        std::cout << ev[i] << ", ";
    }
}

void printEigenVal(Eigen::VectorXf ev){
    std::cout << "\n\n";
    for(int i{0}; i < ev.size(); i++){
        std::cout << ev[i] << ", ";
    }
}

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

    float chosenEigVal = graphInit->eigenValues[findModeIndex(graphInit->eigenValues)];

    printEigenVal(graphInit->eigenValues);

    GradientDescent gdObj(graphInit,weightConstraint,weightSumConstraint);
    gdObj.runNStepDescent(nIterFirst);
    gdObj.destroyHistogram();

    std::vector<double> ev = gdObj.returnEigenvalues();

    printEigenVal(ev);

    int MAX_X = 10;
    int MAX_Y = 10;
    int PLOT_SCALE = 40;
    int vPad = 2;
    int hPad = 2;
    double damping{0.1}, stiffness{5}, epsilon{0.01};
    double amp{1};
    bool decentralizedAlg = false;

    // Generate plot
    Plot my_plot("State Plot - Chosen EigVal: " + std::to_string(chosenEigVal), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
    my_plot.plotGraph(*graphInit);
    my_plot.displayPlot(true);
    
    double freq{sqrt(chosenEigVal)};
    Force my_force(amp, freq, graphInit->nodes.size());
    my_force.insertForceElement(1);


    // Simulate dynamics
    int simulationTime{400};
    int simulationSteps{simulationTime * 100};
    Dynamics my_sim(simulationTime, simulationSteps, damping, stiffness);
    if (decentralizedAlg)
        my_sim.runDecentralizedDynamics(graphInit->nodes, my_force, my_plot);
    else
        my_sim.runCentralizedDynamics(*graphInit, my_force, my_plot);
    my_plot.displayPlot(true);

    return 0;
}
