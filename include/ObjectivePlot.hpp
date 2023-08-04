#ifndef _OBJECTIVEPLOT_H
#define _OBJECTIVEPLOT_H

#include "gnuplot-iostream.h"
#include <vector>

namespace objectivePlot{
    void generateObjectivePlot(Gnuplot &plotObj, std::vector<double> &objectiveValues);
}

#endif
