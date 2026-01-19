#ifndef SEGMENTER_H
#define SEGMENTER_H

#include <string>

class Options;

void runSegmenter(const std::string& in_file, Options& options);

#endif // SEGMENTER_H