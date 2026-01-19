#ifndef ORACLECHILD_H
#define ORACLECHILD_H

struct SegmentInfo {
    size_t start;
    size_t length;
    double hot_cost;
};

int OracleChildMain(int argc, char* argv[]);

#endif