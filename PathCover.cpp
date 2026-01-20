#include "PathCover.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <list>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include "SegmentFile.h"

// --- HELPER STRUCTURES ---
struct Edge {
    size_t to;
    double cost; 
};

// --- LKH INTEGRATION ---

// Write cost matrix in TSPLIB ATSP format
void writeTSPLIB(const std::string& filename, 
                 const std::vector<std::vector<Edge>>& candidates,
                 size_t num_segments) {
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "Failed to write TSPLIB file: " << filename << std::endl;
        return;
    }
    
    f << "NAME: pathcover\n";
    f << "TYPE: ATSP\n";
    f << "DIMENSION: " << num_segments << "\n";
    f << "EDGE_WEIGHT_TYPE: EXPLICIT\n";
    f << "EDGE_WEIGHT_FORMAT: FULL_MATRIX\n";
    f << "EDGE_WEIGHT_SECTION\n";
    
    // Default cost for missing edges
    const int64_t MISSING_EDGE_COST = 999999999;
    
    for (size_t i = 0; i < num_segments; ++i) {
        for (size_t j = 0; j < num_segments; ++j) {
            if (i == j) {
                f << "999999999 ";  // Self-loops forbidden
                continue;
            }
            
            // Look up cost in candidates[i]
            int64_t cost = MISSING_EDGE_COST;
            for (const auto& edge : candidates[i]) {
                if (edge.to == j) {
                    // Costs already normalized to [0, 10000]
                    cost = static_cast<int64_t>(std::round(edge.cost));
                    break;
                }
            }
            
            f << cost << " ";
        }
        f << "\n";
    }
    
    f << "EOF\n";
    f.close();
}

// Write LKH parameter file
void writeLKHParams(const std::string& prob_file, 
                    const std::string& tour_file,
                    const std::string& param_file,
                    int time_limit = 3600) {
    std::ofstream f(param_file);
    if (!f) {
        std::cerr << "Failed to write LKH parameter file: " << param_file << std::endl;
        return;
    }
    
    f << "PROBLEM_FILE = " << prob_file << "\n";
    f << "OUTPUT_TOUR_FILE = " << tour_file << "\n";
    f << "RUNS = 10\n";
    f << "TIME_LIMIT = " << time_limit << "\n";
    f << "TRACE_LEVEL = 1\n";
    f << "SEED = 42\n";
    f << "PRECISION = 10\n";
    
    f.close();
}

// Read LKH tour output
std::vector<size_t> readLKHTour(const std::string& filename, size_t num_segments) {
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "Failed to read LKH tour file: " << filename << std::endl;
        return {};
    }
    
    std::vector<size_t> tour;
    tour.reserve(num_segments);
    
    std::string line;
    bool in_tour_section = false;
    
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        
        if (line.find("TOUR_SECTION") != std::string::npos) {
            in_tour_section = true;
            continue;
        }
        
        if (!in_tour_section) continue;
        
        int node = std::stoi(line);
        if (node == -1) break;
        
        // LKH uses 1-indexed nodes
        tour.push_back(node - 1);
    }
    
    f.close();
    
    if (tour.size() != num_segments) {
        std::cerr << "Warning: Tour size mismatch. Expected " << num_segments 
                  << ", got " << tour.size() << std::endl;
    }
    
    return tour;
}

// --- MAIN ENTRY POINT ---
void runPathCover(const std::string& original_file_base) {
    std::cout << "MCM PathCover: LKH-based Optimization" << std::endl;
    
    // =========================================================
    // 1. LOAD DATA FROM CSV
    // =========================================================
    std::string segments_file = original_file_base + ".segments.csv";
    std::vector<Segment> segments;
    
    try {
        segments = readSegments(segments_file);
        std::cout << "Loaded " << segments.size() << " segments from " << segments_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error reading segments: " << e.what() << std::endl;
        return;
    }
    
    size_t num_segments = segments.size();
    double total_hot_bits = 0;
    for (const auto& seg : segments) {
        total_hot_bits += seg.entropyBits;
    }
    
    std::cout << "Total hot cost: " << total_hot_bits << " bits" << std::endl;
    
    // =========================================================
    // 2. LOAD ORACLE COSTS
    // =========================================================
    std::vector<std::vector<Edge>> candidates(num_segments);
    double total_pretty_warm_bits = 0;
    
    std::string oracle_file = original_file_base + ".oracle";
    std::ifstream oracle_ifs(oracle_file);
    if (!oracle_ifs) {
        std::cerr << "Error opening oracle file: " << oracle_file << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(oracle_ifs, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!std::getline(iss, token, ':')) continue;
        
        size_t pred = std::stoull(token);
        if (pred >= num_segments) continue;
        
        std::string trans_str;
        std::getline(iss, trans_str);
        std::istringstream trans_iss(trans_str);
        
        while (std::getline(trans_iss, token, ';')) {
            if (token.empty()) continue;
            size_t comma = token.find(',');
            if (comma == std::string::npos) continue;
            
            size_t succ = std::stoull(token.substr(0, comma));
            double raw_cost = std::stod(token.substr(comma + 1));
            
            if (succ >= num_segments) continue;
            
            candidates[pred].push_back({succ, raw_cost});
            
            if (succ == pred + 1) {
                total_pretty_warm_bits += (raw_cost + segments[succ].entropyBits);
            }
        }
    }
    oracle_ifs.close();
    
    std::cout << "Loaded oracle costs from " << oracle_file << std::endl;
    
    // =========================================================
    // 3. NORMALIZATION
    // =========================================================
    double warm_to_hot_ratio = total_pretty_warm_bits / total_hot_bits;
    double total_natural_cost_after_normalizing = 0.0;
    std::cout << "Context Scaling Ratio: " << std::fixed << std::setprecision(4) 
              << warm_to_hot_ratio << std::endl;
    
    const double NATURAL_BONUS = 64.0;
    
    for (size_t u = 0; u < candidates.size(); u++) {
        for (auto& edge : candidates[u]) {
            double oracle_delta = edge.cost; 
            double segment_hot_cost = segments[edge.to].entropyBits; 
            double segment_pretty_warm_cost = oracle_delta + segment_hot_cost;
            double segment_estimated_hot_cost = segment_pretty_warm_cost / warm_to_hot_ratio;
            edge.cost = segment_estimated_hot_cost - segment_hot_cost;
            
            if (edge.to == u + 1) {
                total_natural_cost_after_normalizing += edge.cost;
                edge.cost -= NATURAL_BONUS;
            }
        }
    }
    
    std::cout << "Total natural cost after normalizing: " << total_natural_cost_after_normalizing << std::endl;
    
    // Inject natural predecessors
    for (size_t pred = 0; pred < num_segments - 1; ++pred) {
        size_t natural_succ = pred + 1;
        bool found = false;
        for (const auto& edge : candidates[pred]) {
            if (edge.to == natural_succ) {
                found = true;
                break;
            }
        }
        if (!found) {
            double cost = 0.0 - NATURAL_BONUS;
            candidates[pred].push_back({natural_succ, cost});
        }
    }
    
    std::cout << "Injected natural predecessors" << std::endl;
    
    // =========================================================
    // 4. NORMALIZE COST RANGE FOR LKH
    // =========================================================
    double min_cost = std::numeric_limits<double>::max();
    double max_cost = std::numeric_limits<double>::lowest();
    
    for (const auto& candidate_list : candidates) {
        for (const auto& edge : candidate_list) {
            min_cost = std::min(min_cost, edge.cost);
            max_cost = std::max(max_cost, edge.cost);
        }
    }
    
    std::cout << "Cost range before LKH normalization: [" << min_cost << ", " << max_cost << "]" << std::endl;
    
    const double TARGET_MIN = 0.0;
    const double TARGET_MAX = 10000.0;
    double cost_range = max_cost - min_cost;
    
    if (cost_range < 1e-6) {
        std::cerr << "Warning: All costs are identical, using cost_range = 1.0" << std::endl;
        cost_range = 1.0;
    }
    
    for (auto& candidate_list : candidates) {
        for (auto& edge : candidate_list) {
            edge.cost = (edge.cost - min_cost) / cost_range * TARGET_MAX;
        }
    }
    
    std::cout << "Normalized costs to [" << TARGET_MIN << ", " << TARGET_MAX << "] for LKH" << std::endl;
    
    // =========================================================
    // 5. WRITE TSPLIB FILE
    // =========================================================
    std::string tsplib_file = original_file_base + ".atsp";
    std::cout << "Writing TSPLIB file: " << tsplib_file << std::endl;
    writeTSPLIB(tsplib_file, candidates, num_segments);
    
    // =========================================================
    // 6. WRITE LKH PARAMETER FILE
    // =========================================================
    std::string tour_file = original_file_base + ".tour";
    std::string param_file = original_file_base + ".par";
    
    std::cout << "Writing LKH parameter file: " << param_file << std::endl;
    writeLKHParams(tsplib_file, tour_file, param_file, 3600);
    
    // =========================================================
    // 7. RUN LKH
    // =========================================================
    std::cout << "\n=== Running LKH ===" << std::endl;
    std::string lkh_cmd = "LKH " + param_file;
    
    int ret = system(lkh_cmd.c_str());
    if (ret != 0) {
        std::cerr << "LKH failed with return code: " << ret << std::endl;
        std::cerr << "Make sure LKH executable is in PATH or current directory" << std::endl;
        return;
    }
    
    std::cout << "LKH completed successfully" << std::endl;
    
    // =========================================================
    // 8. READ LKH TOUR
    // =========================================================
    std::cout << "\nReading LKH tour from: " << tour_file << std::endl;
    std::vector<size_t> order = readLKHTour(tour_file, num_segments);
    
    if (order.empty() || order.size() != num_segments) {
        std::cerr << "Failed to read valid tour from LKH" << std::endl;
        return;
    }
    
    std::cout << "Tour size: " << order.size() << std::endl;
    
    // =========================================================
    // 9. VERIFY TOUR
    // =========================================================
    std::vector<bool> seen(num_segments, false);
    for (size_t seg : order) {
        if (seg >= num_segments) {
            std::cerr << "Invalid segment ID in tour: " << seg << std::endl;
            return;
        }
        if (seen[seg]) {
            std::cerr << "Duplicate segment in tour: " << seg << std::endl;
            return;
        }
        seen[seg] = true;
    }
    
    std::cout << "Tour verification: PASSED" << std::endl;
    
    // =========================================================
    // 10. EXPORT INDEX FILE
    // =========================================================
    std::string index_file = original_file_base + ".index";
    std::ofstream idx_ofs(index_file);
    if (!idx_ofs) {
        std::cerr << "Failed to write index file: " << index_file << std::endl;
        return;
    }
    
    for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) idx_ofs << " ";
        idx_ofs << order[i];
    }
    idx_ofs.close();
    
    std::cout << "Wrote segment order to " << index_file << std::endl;
    
    // =========================================================
    // 11. EXPORT REORDERED FILE
    // =========================================================
    std::string original_file = original_file_base;
    std::ifstream data_ifs(original_file, std::ios::binary | std::ios::ate);
    if (!data_ifs) {
        std::cerr << "Failed to open original file: " << original_file << std::endl;
        return;
    }
    
    size_t file_size = data_ifs.tellg();
    data_ifs.seekg(0);
    
    std::vector<uint8_t> file_data(file_size);
    data_ifs.read(reinterpret_cast<char*>(file_data.data()), file_size);
    data_ifs.close();
    
    std::string reordered_file = original_file_base + ".reordered";
    std::ofstream rofs(reordered_file, std::ios::binary);
    if (!rofs) {
        std::cerr << "Failed to create reordered file: " << reordered_file << std::endl;
        return;
    }
    
    for (size_t seg_idx : order) {
        const auto& seg = segments[seg_idx];
        
        if (seg.startByte >= file_size) {
            std::cerr << "Error: Segment " << seg_idx << " start beyond file size" << std::endl;
            continue;
        }
        
        size_t len = seg.lengthBytes;
        if (seg.startByte + len > file_size) {
            len = file_size - seg.startByte;
        }
        
        rofs.write(reinterpret_cast<const char*>(&file_data[seg.startByte]), len);
    }
    
    rofs.close();
    std::cout << "Wrote reordered file to " << reordered_file << std::endl;
    
    // =========================================================
    // 12. UPDATE SEGMENTS CSV
    // =========================================================
    std::vector<size_t> inverse_order(num_segments);
    for (size_t pos = 0; pos < order.size(); ++pos) {
        inverse_order[order[pos]] = pos;
    }
    
    for (size_t i = 0; i < segments.size(); ++i) {
        segments[i].reorderedIndex = inverse_order[i];
        
        if (segments[i].phaseCompleted.find("pathcover") == std::string::npos) {
            if (!segments[i].phaseCompleted.empty()) {
                segments[i].phaseCompleted += ",";
            }
            segments[i].phaseCompleted += "pathcover";
        }
    }
    
    try {
        writeSegments(segments_file, segments);
        std::cout << "Updated segments CSV with reordered indices" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error writing updated segments: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== PathCover Complete ===" << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "  1. Compress " << reordered_file << " with MCM" << std::endl;
    std::cout << "  2. Compare size to original compression" << std::endl;
}