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
            std::vector<std::string> expected_latest = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropyBitsPerByte", "entropySpike",
                "fingerprint", "aloneEntropyBits", "aloneEntropyBitsPerByte", "entropyBytes256", "entropyBytes2048",
                "reorderedIndex", "predictedReorderedEntropyBits", "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            std::vector<std::string> expected_new = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropyBitsPerByte", "entropySpike",
                "fingerprint", "aloneEntropyBits", "aloneEntropyBitsPerByte", "reorderedIndex", "predictedReorderedEntropyBits",
                "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            std::vector<std::string> expected_inter = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropyBitsPerByte", "entropySpike",
                "fingerprint", "aloneEntropyBits", "reorderedIndex", "predictedReorderedEntropyBits",
                "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            std::vector<std::string> expected_old = {
                "index", "startByte", "lengthBytes", "entropyBits", "entropySpike",
                "fingerprint", "aloneEntropyBits", "reorderedIndex", "predictedReorderedEntropyBits",
                "calculatedReorderedEntropyBit", "phaseCompleted"
            };
            bool isLatest = (fields.size() == expected_latest.size());
            bool isNew = (fields.size() == expected_new.size());
            bool isInter = (fields.size() == expected_inter.size());
            bool isOld = (fields.size() == expected_old.size());
            if (!isLatest && !isNew && !isInter && !isOld) {
                throw std::runtime_error("Invalid CSV header in " + filename + ": expected 11, 12, 13, or 15 fields, got " + std::to_string(fields.size()));
            }
            const std::vector<std::string>& expected = isLatest ? expected_latest : (isNew ? expected_new : (isInter ? expected_inter : expected_old));
            for (size_t i = 0; i < expected.size(); ++i) {
                if (fields[i] != expected[i]) {
                    throw std::runtime_error("Header mismatch in " + filename + ": expected " + expected[i] + ", got " + fields[i]);
                }
            }
            isHeader = false;
            continue;
        }

        // Parse data row
        bool isLatest = (fields.size() == 15);
        bool isNew = (fields.size() == 13);
        bool isInter = (fields.size() == 12);
        bool isOld = (fields.size() == 11);
        if (!isLatest && !isNew && !isInter && !isOld) {
            throw std::runtime_error("Invalid row in " + filename + ": expected 11, 12, 13, or 15 fields, got " + std::to_string(fields.size()));
        }

        Segment seg;
        try {
            seg.index = std::stoull(fields[0]);
            seg.startByte = std::stoull(fields[1]);
            seg.lengthBytes = std::stoull(fields[2]);
            seg.entropyBits = std::stod(fields[3]);
            if (isLatest) {
                seg.entropyBitsPerByte = std::stod(fields[4]);
                seg.entropySpike = std::stod(fields[5]);
                seg.fingerprint = fields[6];
                seg.aloneEntropyBits = std::stod(fields[7]);
                seg.aloneEntropyBitsPerByte = std::stod(fields[8]);
                seg.entropyBytes256 = std::stod(fields[9]);
                seg.entropyBytes2048 = std::stod(fields[10]);
                if (fields[11] != "" && fields[11] != "-1") seg.reorderedIndex = std::stoull(fields[11]);
                seg.predictedReorderedEntropyBits = std::stod(fields[12]);
                seg.calculatedReorderedEntropyBit = std::stod(fields[13]);
                seg.phaseCompleted = fields[14];
            } else if (isNew) {
                seg.entropyBitsPerByte = std::stod(fields[4]);
                seg.entropySpike = std::stod(fields[5]);
                seg.fingerprint = fields[6];
                seg.aloneEntropyBits = std::stod(fields[7]);
                seg.aloneEntropyBitsPerByte = std::stod(fields[8]);
                seg.entropyBytes256 = 0.0;  // default
                seg.entropyBytes2048 = 0.0; // default
                if (fields[9] != "" && fields[9] != "-1") seg.reorderedIndex = std::stoull(fields[9]);
                seg.predictedReorderedEntropyBits = std::stod(fields[10]);
                seg.calculatedReorderedEntropyBit = std::stod(fields[11]);
                seg.phaseCompleted = fields[12];
            } else if (isInter) {  // 12 fields, has entropyBitsPerByte but not alone
                seg.entropyBitsPerByte = std::stod(fields[4]);
                seg.entropySpike = std::stod(fields[5]);
                seg.fingerprint = fields[6];
                seg.aloneEntropyBits = std::stod(fields[7]);
                seg.aloneEntropyBitsPerByte = 0.0;  // default
                seg.entropyBytes256 = 0.0;  // default
                seg.entropyBytes2048 = 0.0; // default
                if (fields[8] != "" && fields[8] != "-1") seg.reorderedIndex = std::stoull(fields[8]);
                seg.predictedReorderedEntropyBits = std::stod(fields[9]);
                seg.calculatedReorderedEntropyBit = std::stod(fields[10]);
                seg.phaseCompleted = fields[11];
            } else {  // old format 11
                seg.entropyBitsPerByte = seg.entropyBits / seg.lengthBytes; // compute
                seg.entropySpike = std::stod(fields[4]);
                seg.fingerprint = fields[5];
                seg.aloneEntropyBits = std::stod(fields[6]);
                seg.aloneEntropyBitsPerByte = 0.0; // default
                seg.entropyBytes256 = 0.0;  // default
                seg.entropyBytes2048 = 0.0; // default
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
    file << "index,startByte,lengthBytes,entropyBits,entropyBitsPerByte,entropySpike,fingerprint,aloneEntropyBits,aloneEntropyBitsPerByte,entropyBytes256,entropyBytes2048,reorderedIndex,predictedReorderedEntropyBits,calculatedReorderedEntropyBit,phaseCompleted\n";

    // Write data
    for (const auto& seg : segments) {
        file << seg.index << ","
             << seg.startByte << ","
             << seg.lengthBytes << ","
             << seg.entropyBits << ","
             << seg.entropyBitsPerByte << ","
             << seg.entropySpike << ","
             << escapeCSV(seg.fingerprint) << ","
             << seg.aloneEntropyBits << ","
             << seg.aloneEntropyBitsPerByte << ","
             << seg.entropyBytes256 << ","
             << seg.entropyBytes2048 << ",";
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