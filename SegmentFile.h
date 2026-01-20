#ifndef SEGMENTFILE_H
#define SEGMENTFILE_H

#include <vector>
#include <string>
#include <cstdint>

struct Segment {
    size_t index = 0;
    size_t startByte = 0;
    size_t lengthBytes = 0;
    double entropyBits = 0.0;
    double entropyBitsPerByte = 0.0;
    double entropySpike = 0.0;
    std::string fingerprint = "";
    double aloneEntropyBits = 0.0;
    double aloneEntropyBitsPerByte = 0.0;
    double entropyBytes256 = 0.0;  // Hot cost at 256 bytes
    double entropyBytes2048 = 0.0; // Hot cost at 2048 bytes
    size_t reorderedIndex = static_cast<size_t>(-1);  // -1 means unset
    double predictedReorderedEntropyBits = 0.0;
    double calculatedReorderedEntropyBit = 0.0;  // Note: singular "Bit" as per your spec
    std::string phaseCompleted = "";  // e.g., "entropy,fingerprint,alone,reorder"

    // Helper to check if required fields are set
    bool isValid() const {
        return lengthBytes > 0;  // Basic check; expand as needed
    }

    // Reset optional fields
    void resetOptional() {
        entropyBitsPerByte = 0.0;
        entropySpike = 0.0;
        fingerprint = "";
        aloneEntropyBits = 0.0;
        aloneEntropyBitsPerByte = 0.0;
        entropyBytes256 = 0.0;
        entropyBytes2048 = 0.0;
        reorderedIndex = static_cast<size_t>(-1);
        predictedReorderedEntropyBits = 0.0;
        calculatedReorderedEntropyBit = 0.0;
        phaseCompleted = "";
    }
};

std::vector<Segment> readSegments(const std::string& filename);
void writeSegments(const std::string& filename, const std::vector<Segment>& segments);

#endif