#include "ObjectivePlot.hpp"
#include <numeric>
#include <vector>

namespace objectivePlot{
    void generateObjectivePlot(Gnuplot &plotObj, std::vector<double> &objectiveValues){

        double minVal = *std::min_element(objectiveValues.begin(), objectiveValues.end());
        double maxVal = *std::max_element(objectiveValues.begin(), objectiveValues.end());

        plotObj << "reset\n";
        plotObj << "min=" << minVal << "\n";
        plotObj << "max=" << maxVal << "\n";

        plotObj << "set yrange [min:max]\n";
        plotObj << "set xrange [0:" << objectiveValues.size()+1 << "]\n";

        plotObj << "set xlabel 'Iteration'\n";
        plotObj << "set ylabel 'Objective Value'\n";


        plotObj << "plot '-' u 1:2 with lines lc rgb'green' notitle\n"; 

        for (size_t i = 0; i < objectiveValues.size(); ++i) {
            plotObj << i << " " << objectiveValues[i] << "\n";
        }

        plotObj << "e\n";

        plotObj.flush();
        usleep(10000);
    }
}
