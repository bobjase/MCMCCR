#include "SegmentFile.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// Helper to escape CSV fields (add quotes if needed)
std::string escapeCSV(const std::string& field) {
    if (field.find(',') != std::string::npos || field.find('"') != std::string::npos) {
        std::string escaped = field;
        // Replace " with ""
        size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos) {
            escaped.replace(pos, 1, "\"\"");
            pos += 2;
        }
        return "\"" + escaped + "\"";
    }
    return field;
}

// Helper to parse a CSV field (handle quotes)
std::string parseCSVField(const std::string& field) {
    if (field.empty()) return "";
    if (field.front() == '"' && field.back() == '"') {
        std::string unescaped = field.substr(1, field.size() - 2);
        // Replace "" with "
        size_t pos = 0;
        while ((pos = unescaped.find("\"\"", pos)) != std::string::npos) {
            unescaped.replace(pos, 2, "\"");
            pos += 1;
        }
        return unescaped;
    }
    return field;
}

std::vector<Segment> readSegments(const std::string& filename) {
    std::vector<Segment> segments;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    bool isHeader = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> fields;

        // Parse CSV line
        while (std::getline(iss, token, ',')) {
            // Handle quoted fields spanning multiple commas (basic handling)
            if (!token.empty() && token.front() == '"' && token.back() != '"') {
                std::string next;
                while (std::getline(iss, next, ',')) {
                    token += "," + next;
                    if (next.back() == '"') break;
                }
            }
            fields.push_back(parseCSVField(token));
        }

        if (isHeader) {
            // Check header
            std::vector<std::string> expected_new = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropyBitsPerByte", "entropySpike",
                "fingerprint", "aloneEntropyBits", "reorderedIndex", "predictedReorderedEntropyBits",
                "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            std::vector<std::string> expected_old = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropySpike",
                "fingerprint", "aloneEntropyBits", "reorderedIndex", "predictedReorderedEntropyBits",
                "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            bool isNewFormat = (fields.size() == expected_new.size());
            bool isOldFormat = (fields.size() == expected_old.size());
            if (!isNewFormat && !isOldFormat) {
                throw std::runtime_error("Invalid CSV header in " + filename + ": expected 11 or 12 fields, got " + std::to_string(fields.size()));
            }
            if (isNewFormat) {
                for (size_t i = 0; i < expected_new.size(); ++i) {
                    if (fields[i] != expected_new[i]) {
                        throw std::runtime_error("Header mismatch in " + filename + ": expected " + expected_new[i] + ", got " + fields[i]);
                    }
                }
            } else { // old format
                for (size_t i = 0; i < expected_old.size(); ++i) {
                    if (fields[i] != expected_old[i]) {
                        throw std::runtime_error("Header mismatch in " + filename + ": expected " + expected_old[i] + ", got " + fields[i]);
                    }
                }
            }
            isHeader = false;
            continue;
        }

        // Parse data row
        bool isNewFormat = (fields.size() == 12);
        if (fields.size() != 11 && fields.size() != 12) {
            throw std::runtime_error("Invalid row in " + filename + ": expected 11 or 12 fields, got " + std::to_string(fields.size()));
        }

        Segment seg;
        try {
            seg.index = std::stoull(fields[0]);
            seg.startByte = std::stoull(fields[1]);
            seg.lengthBytes = std::stoull(fields[2]);
            seg.entropyBits = std::stod(fields[3]);
            if (isNewFormat) {
                seg.entropyBitsPerByte = std::stod(fields[4]);
                seg.entropySpike = std::stod(fields[5]);
                seg.fingerprint = fields[6];
                seg.aloneEntropyBits = std::stod(fields[7]);
                if (fields[8] != "" && fields[8] != "-1") seg.reorderedIndex = std::stoull(fields[8]);
                seg.predictedReorderedEntropyBits = std::stod(fields[9]);
                seg.calculatedReorderedEntropyBit = std::stod(fields[10]);
                seg.phaseCompleted = fields[11];
            } else {
                seg.entropyBitsPerByte = seg.entropyBits / seg.lengthBytes; // compute for old format
                seg.entropySpike = std::stod(fields[4]);
                seg.fingerprint = fields[5];
                seg.aloneEntropyBits = std::stod(fields[6]);
                if (fields[7] != "" && fields[7] != "-1") seg.reorderedIndex = std::stoull(fields[7]);
                seg.predictedReorderedEntropyBits = std::stod(fields[8]);
                seg.calculatedReorderedEntropyBit = std::stod(fields[9]);
                seg.phaseCompleted = fields[10];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Parse error in " + filename + ": " + e.what());
        }

        segments.push_back(seg);
    }

    return segments;
}

void writeSegments(const std::string& filename, const std::vector<Segment>& segments) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write header
    file << "index,startByte,lengthBytes,entropyBits,entropyBitsPerByte,entropySpike,fingerprint,aloneEntropyBits,reorderedIndex,predictedReorderedEntropyBits,calculatedReorderedEntropyBit,phaseCompleted\n";

    // Write data
    for (const auto& seg : segments) {
        file << seg.index << ","
             << seg.startByte << ","
             << seg.lengthBytes << ","
             << seg.entropyBits << ","
             << seg.entropyBitsPerByte << ","
             << seg.entropySpike << ","
             << escapeCSV(seg.fingerprint) << ","
             << seg.aloneEntropyBits << ",";
        if (seg.reorderedIndex == static_cast<size_t>(-1)) {
            file << ",";
        } else {
            file << seg.reorderedIndex << ",";
        }
        file << seg.predictedReorderedEntropyBits << ","
             << seg.calculatedReorderedEntropyBit << ","
             << escapeCSV(seg.phaseCompleted) << "\n";
    }
}