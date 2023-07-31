SHELL := /bin/bash

# Change these if needed or pass as input parameter
# PREFIX = /usr/local

# ================================

INC_LOCAL = -I. -I..

# --------------------------------------------
# Macros

_BOLD = "\\033[1m"
_DIM = "\\033[2m"
_RED = "\\033[31m"
_GREEN = "\\033[32m"
_YELLOW = "\\033[33m"
_BLUE ="\\033[34m"
_GRAY = "\\033[37m"
_CLEAR = "\\033[0m"

COMPILATION_SUCCESS_TEXT = "$(_GREEN)%s compilation successful!$(_CLEAR)\n"
COMPILATION_FAILURE_TEXT = "$(_RED)$(_BOLD)ERROR:$(_CLEAR) $(_RED)%s compilation failed!$(_CLEAR)\n"

COMPILE_PROGRAM = printf "$(_DIM)$(subst `,\`,$(strip $(2)))$(_CLEAR)\n" ; if $(2) ; then printf $(COMPILATION_SUCCESS_TEXT) "$(strip $(1))" ; else printf $(COMPILATION_FAILURE_TEXT) "$(strip $(1))" ; fi

# --------------------------------------------
# Dependencies


# --------------------------------------------
# common flags

CC = g++
CFLAGS = -std=gnu++17 -g -O3
PROFILE = -pg
WARNS = -w

LIBS = -lm -pthread
LIBS_OPENCV = `pkg-config --cflags --libs opencv` -L/usr/local/share/OpenCV/3rdparty/lib/  # The -L option fixes bug with libippicv
LIBS_GNUPLOT = -lboost_iostreams -lboost_system -lboost_filesystem

LIB_FOLDERS = -L/usr/local/lib

BIN_FOLDER = bin
SRC_FOLDER = src
MAIN_HEADERS = include/Plot.cpp include/Graph.cpp include/Node.cpp include/Dynamics.cpp include/Force.cpp include/Histogram.cpp include/GradientDescent.cpp

.PHONY: gdMain
gdMain:
	@printf "\nNow compiling '$(_BLUE)$@$(_CLEAR)'...\n";
	@$(call COMPILE_PROGRAM,\
		$@,\
		$(CC) $(CFLAGS) $(WARNS) $(INC_LOCAL) -o $(BIN_FOLDER)/$@ $(SRC_FOLDER)/$@.cpp $(MAIN_HEADERS) $(LIB_FOLDERS) $(LIBS) $(LIBS_OPENCV) $(LIBS_GNUPLOT)\
	)
