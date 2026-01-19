#include "Segmenter.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include "File.hpp"
#include "EntropySegmenter.h"

// Forward declare Options if needed, but since it's included via MCM.cpp linkage, perhaps not.
// Since Options is in MCM.cpp, and we're linking, it should be fine, but to be safe, I'll assume it's available.

static void printHeader() {
    std::cout << "MCM file compressor" << std::endl;
}

void runSegmenter(const std::string& in_file, Options& options) {
    printHeader();
    std::cout << "Running Auto-Tuned PELT Segmentation (Macro-Micro)" << std::endl;

    std::string entropy_file = in_file + ".entropy";

    // 1. Load Entropy
    std::ifstream fin_ent(entropy_file, std::ios::binary);
    if (!fin_ent) { std::cerr << "Error: Missing .entropy file." << std::endl; return; }
    uint64_t num_bytes;
    fin_ent.read((char*)&num_bytes, sizeof(num_bytes));
    if (!fin_ent) { std::cerr << "Error: Failed to read entropy file header." << std::endl; return; }
    uint64_t stock_size;
    fin_ent.read((char*)&stock_size, sizeof(stock_size));
    if (!fin_ent) { std::cerr << "Error: Failed to read entropy file header." << std::endl; return; }
    std::vector<double> full_entropy(num_bytes);
    fin_ent.read((char*)full_entropy.data(), num_bytes * sizeof(double));
    if (!fin_ent) { std::cerr << "Error: Failed to read entropy data." << std::endl; return; }
    fin_ent.close();

    // 2. SPEED FIX: Downsample (Coarse-Graining)
    // Averages every X bytes into 1 point. 
    // Reduces massive number of points -> ~500,000 points. Runs fast even on large files.
    const size_t DOWNSAMPLE_RATE = std::max(size_t(1), size_t(std::floor(num_bytes / 500000.0)));
    std::vector<double> coarse_profile;
    coarse_profile.reserve(num_bytes / DOWNSAMPLE_RATE + 1);

    for (size_t i = 0; i < full_entropy.size(); i += DOWNSAMPLE_RATE) {
        double sum = 0;
        size_t count = 0;
        for (size_t k = 0; k < DOWNSAMPLE_RATE && (i+k) < full_entropy.size(); ++k) {
            sum += full_entropy[i+k];
            count++;
        }
        coarse_profile.push_back(sum / count); 
    }
    std::cout << "Downsampled profile: " << full_entropy.size() << " -> " << coarse_profile.size() << " points." << std::endl;

    // 3. Run PELT on Coarse Data
    // Scale penalty for smaller dataset size
    // Base penalty logic: ~20 * log(N)
    double raw_penalty = 50.0 * std::log(coarse_profile.size());
    
    // Scale it down to match the downsampled signal energy
    double penalty = raw_penalty / (double)DOWNSAMPLE_RATE;
    
    // Safety floor: Don't let penalty hit 0 or cuts become free
    if (penalty < 0.1) penalty = 0.1;

    // Minimum 1 coarse block (represents 64 real bytes)
    size_t min_coarse_size = 4096 / DOWNSAMPLE_RATE; 
    if (min_coarse_size < 1) min_coarse_size = 1;

    std::cout << "Analyzing Structure (PELT)... Penalty=" << penalty << std::endl;
    std::vector<size_t> coarse_cuts = EntropySegmenter::FindCuts(coarse_profile, penalty, min_coarse_size);
    
    // 4. Scale & Snap to "Model Peaks" (High Precision Fix)
    // We use the Coarse PELT to find the neighborhood, then scan the 
    // Full Resolution profile to find the exact byte where confusion spiked.
    std::vector<size_t> boundaries;
    boundaries.push_back(0);
    
    // Window to search for the spike (covers the downsampling blur)
    const size_t SNAP_RADIUS = DOWNSAMPLE_RATE * 2; 

    for (size_t coarse_cut : coarse_cuts) {
        size_t approx_loc = coarse_cut * DOWNSAMPLE_RATE;
        
        // Don't snap 0 or EOF
        if (approx_loc == 0 || approx_loc >= full_entropy.size()) continue;

        // Define search window in full_entropy
        size_t start = (approx_loc > SNAP_RADIUS) ? approx_loc - SNAP_RADIUS : 0;
        size_t end = std::min(full_entropy.size(), approx_loc + SNAP_RADIUS);

        // Find the "Surprisal Cliff" (Local Max Entropy)
        // The semantic boundary is usually where the model is MOST confused.
        double max_val = -1.0;
        size_t best_idx = approx_loc;

        for (size_t i = start; i < end; ++i) {
            // Optional: Use a simple 3-byte moving average to ignore single-byte noise
            // but for now, raw max is usually the header start.
            if (full_entropy[i] > max_val) {
                max_val = full_entropy[i];
                best_idx = i;
            }
        }
        
        boundaries.push_back(best_idx);
    }
    
    // Ensure file end
    std::ifstream fin_check(in_file, std::ios::binary | std::ios::ate);
    size_t file_size = fin_check.tellg();
    fin_check.close();
    
    // Sort and dedup just in case snapping overlapped
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    
    if (boundaries.back() < file_size) boundaries.push_back(file_size);

    std::cout << "PELT found " << boundaries.size() - 1 << " regions (Snapped to Peak Entropy)." << std::endl;

    // // 5. Boundary Refinement Phase ("Inductive Boundary Snap")
    // // (This contains your recent Crash Fix)
    // std::cout << "Refining " << boundaries.size() << " boundaries..." << std::endl;
    // std::string original_file = in_file;
    // std::ifstream fin_refine(original_file, std::ios::binary);
    // if (fin_refine) {
    //   const size_t BLOCK_SIZE = (options.segment_window > 0) ? options.segment_window : 512;
    //   const size_t SEARCH_RADIUS = BLOCK_SIZE / 2;
      
    //   const size_t LOG_TABLE_SIZE = 65536; 
    //   std::vector<double> x_log_x(LOG_TABLE_SIZE);
    //   x_log_x[0] = 0.0;
    //   for (size_t i = 1; i < LOG_TABLE_SIZE; ++i) x_log_x[i] = i * std::log2(i);

    //   for (size_t i = 1; i < boundaries.size() - 1; ++i) {
    //     size_t coarse_cut = boundaries[i];
    //     size_t search_start = (coarse_cut > SEARCH_RADIUS) ? coarse_cut - SEARCH_RADIUS : 0;
    //     size_t search_end = std::min(file_size, coarse_cut + SEARCH_RADIUS);
        
    //     if (search_start <= boundaries[i - 1]) search_start = boundaries[i - 1] + 1;
    //     if (search_end >= boundaries[i + 1]) search_end = boundaries[i + 1] - 1;
        
    //     // Safety Check (The Fix)
    //     if (search_end <= search_start) continue;

    //     size_t window_len = search_end - search_start;
    //     if (window_len < 2) continue;
    //     std::vector<uint8_t> window(window_len);
    //     fin_refine.seekg(search_start);
    //     fin_refine.read(reinterpret_cast<char*>(window.data()), window_len);

    //     std::array<uint32_t, 256> right_counts = {0};
    //     std::array<uint32_t, 256> left_counts = {0};
    //     double right_sum = 0.0;
    //     double left_sum = 0.0;

    //     for (uint8_t b : window) {
    //       if (right_counts[b] > 0) right_sum -= x_log_x[right_counts[b]];
    //       right_counts[b]++;
    //       right_sum += x_log_x[right_counts[b]];
    //     }

    //     double min_local_cost = std::numeric_limits<double>::infinity();
    //     size_t best_local_offset = 0;

    //     for (size_t k = 0; k < window_len - 1; ++k) {
    //       uint8_t b = window[k];
    //       right_sum -= x_log_x[right_counts[b]];
    //       right_counts[b]--;
    //       if (right_counts[b] > 0) right_sum += x_log_x[right_counts[b]];

    //       if (left_counts[b] > 0) left_sum -= x_log_x[left_counts[b]];
    //       left_counts[b]++;
    //       left_sum += x_log_x[left_counts[b]];

    //       size_t len_l = k + 1;
    //       size_t len_r = window_len - len_l;
    //       double cost_l = (len_l < LOG_TABLE_SIZE ? x_log_x[len_l] : len_l * std::log2(len_l)) - left_sum;
    //       double cost_r = (len_r < LOG_TABLE_SIZE ? x_log_x[len_r] : len_r * std::log2(len_r)) - right_sum;
          
    //       if (cost_l + cost_r < min_local_cost) {
    //         min_local_cost = cost_l + cost_r;
    //         best_local_offset = k + 1;
    //       }
    //     }
    //     boundaries[i] = search_start + best_local_offset;
    //   }
    //   fin_refine.close();
    //   std::cout << "Refinement Complete." << std::endl;
    // }

    // ... [Previous Steps 1-5 remain exactly the same] ...

    // 6. CALCULATE HOT COSTS FROM EXISTING ENTROPY DATA
    std::cout << "Calculating Natural (Hot) Costs from .entropy vector..." << std::endl;

    struct FinalSegmentInfo {
        size_t start;
        size_t len;
        double hot_cost;
    };
    std::vector<FinalSegmentInfo> final_segments;

    for (size_t i = 0; i < boundaries.size() - 1; ++i) {
        size_t start = boundaries[i];
        size_t len = boundaries[i+1] - boundaries[i];
        
        // Sum the entropy values for this specific segment's range
        double segment_hot_sum = 0.0;
        
        // Safety check to stay within vector bounds
        size_t end_idx = std::min(start + len, full_entropy.size());
        for (size_t k = start; k < end_idx; ++k) {
            segment_hot_sum += full_entropy[k];
        }

        final_segments.push_back({start, len, segment_hot_sum});
    }

    // 7. Write Output (Start, Length, HotCost)
    std::string segments_out_file = in_file + ".segments";
    std::ofstream ofs(segments_out_file);
    if (!ofs) { std::cerr << "Error: Could not write .segments file." << std::endl; return; }

    ofs << final_segments.size() << "\n";
    for (const auto& seg : final_segments) {
        // Use fixed and precision for stable parsing in the Oracle Child
        ofs << seg.start << " " 
            << seg.len << " " 
            << std::fixed << std::setprecision(4) << seg.hot_cost << "\n";
    }
    ofs.close();

    std::cout << "Wrote " << final_segments.size() << " segments with Hot Costs to " << segments_out_file << std::endl;
}