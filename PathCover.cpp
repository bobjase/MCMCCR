#include "PathCover.h"
#include "csa.hpp" 

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
#include <random>
#include <cstdlib> 

// --- 1. DEFINE EDGE FIRST ---
struct Edge {
    size_t to;
    double cost; 
};

// --- 2. DEFINE DATA STRUCTURES ---
struct SegmentInfo {
    size_t start;
    size_t length;
    double hot_cost;
};

// Passed to CSA callbacks
struct ProblemData {
    size_t num_segments;
    std::vector<std::vector<Edge>>* candidates;
    
    inline double get_cost(int u, int v) const {
        if (u < 0 || v < 0 || u >= (int)num_segments || v >= (int)num_segments) return 0.0;
        for (const auto& e : (*candidates)[u]) {
            if ((int)e.to == v) return e.cost;
        }
        return 0.0; 
    }
};

// --- CSA CALLBACKS ---

// Cost Function (FIXED: Detects Lassos)
double csa_cost_function(void* instance, int* next_seg_array) {
    ProblemData* data = (ProblemData*)instance;
    size_t N = data->num_segments;
    
    double total_cost = 0.0;
    int visited_count = 0;
    
    // A. Sum Transition Costs
    for (size_t i = 0; i < N; ++i) {
        if (next_seg_array[i] != -1) {
            total_cost += data->get_cost(i, next_seg_array[i]);
        }
    }

    // B. Cycle & Lasso Detection
    std::vector<int> indegree(N, 0);
    for (size_t i = 0; i < N; ++i) {
        if (next_seg_array[i] != -1) indegree[next_seg_array[i]]++;
    }
    
    std::vector<bool> visited(N, false);
    for (size_t i = 0; i < N; ++i) {
        if (indegree[i] == 0) { // Found a Head
            int curr = i;
            int safety = 0;
            bool clean_end = false;
            
            // Traverse the chain
            while (curr != -1) {
                if (visited[curr]) {
                    // We hit a node we already visited THIS TRAVERSAL or PREVIOUSLY.
                    // This means we found a LASSO (6->7->8->7).
                    // We fail to set clean_end.
                    break; 
                }
                visited[curr] = true;
                visited_count++;
                curr = next_seg_array[curr];
                
                if (curr == -1) clean_end = true; // Properly terminated
                if (++safety > (int)N * 2) break; 
            }
            
            // If the chain didn't end in -1, it's a lasso. Penalize.
            if (!clean_end) {
                total_cost += 1e16; // Massive penalty for Lasso
            }
        }
    }
    
    // Any node not visited is part of a perfect Ring (unreachable).
    if (visited_count < (int)N) {
        total_cost += (N - visited_count) * 1e15; 
    }

    return total_cost;
}

// Step Function (Mutation)
void csa_step_function(void* instance, int* y, const int* x, float tgen) {
    ProblemData* data = (ProblemData*)instance;
    size_t N = data->num_segments;
    
    std::copy(x, x + N, y);

    auto get_rand_01 = []() { return (double)rand() / (double)RAND_MAX; };

    int num_moves = 1;
    if (tgen > 0.5) num_moves = 3;
    if (tgen > 0.1 && get_rand_01() < 0.5) num_moves = 2;

    for (int k = 0; k < num_moves; ++k) {
        double op = get_rand_01();
        size_t u = rand() % N;

        // OP A: THE KNIFE (Cut)
        if (op < 0.30) {
            if (y[u] != -1) {
                y[u] = -1;
            }
        }
        // OP B: 2-OPT (Swap Targets)
        else {
            size_t v = rand() % N;
            if (u != v) {
                int u_next = y[u];
                int v_next = y[v];
                
                y[u] = v_next;
                y[v] = u_next;
                
                if (y[u] == (int)u) y[u] = u_next; 
                if (y[v] == (int)v) y[v] = v_next; 
            }
        }
    }
}

// Progress Callback
void csa_progress(void* instance, double cost, float tgen, float tacc, int opt_id, int iter) {
    if (iter % 1000 == 0) {
        std::cout << "\rStep " << iter/1000 << "k | Cost: " << (long)cost 
                  << " | T_gen: " << std::scientific << std::setprecision(2) << tgen 
                  << " | Thread: " << opt_id << "   " << std::flush;
    }
}

// --- MAIN ENTRY POINT ---
void runPathCover(const std::string& original_file_base) {
    std::cout << "MCM PathCover: Coupled Simulated Annealing (CSA)" << std::endl;
    
    // =========================================================
    // 1. LOAD DATA
    // =========================================================
    std::ifstream seg_ifs(original_file_base + ".segments");
    size_t num_segments;
    seg_ifs >> num_segments;
    std::vector<SegmentInfo> segments(num_segments);
    double total_hot_bits = 0;
    for (size_t i = 0; i < num_segments; ++i) {
        seg_ifs >> segments[i].start >> segments[i].length >> segments[i].hot_cost;
        total_hot_bits += segments[i].hot_cost;
    }
    seg_ifs.close();

    std::vector<std::vector<Edge>> candidates(num_segments);
    double total_pretty_warm_bits = 0;
    
    std::ifstream oracle_ifs(original_file_base + ".oracle");
    std::string line;
    while (std::getline(oracle_ifs, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!std::getline(iss, token, ':')) continue;
        size_t pred = std::stoul(token);
        std::string trans_str;
        std::getline(iss, trans_str);
        std::istringstream trans_iss(trans_str);
        while (std::getline(trans_iss, token, ';')) {
            if (token.empty()) continue;
            size_t comma = token.find(',');
            size_t succ = std::stoul(token.substr(0, comma));
            double raw_cost = std::stod(token.substr(comma + 1));
            candidates[pred].push_back({succ, raw_cost});
            if (succ == pred + 1) {
                total_pretty_warm_bits += (raw_cost + segments[succ].hot_cost);
            }
        }
    }
    oracle_ifs.close();

    // =========================================================
    // 2. NORMALIZATION
    // =========================================================
    double warm_to_hot_ratio = total_pretty_warm_bits / total_hot_bits;
    double total_natural_cost_after_normalizing_should_be_zero = 0.0;
    std::cout << "Context Scaling Ratio: " << std::fixed << std::setprecision(4) << warm_to_hot_ratio << std::endl;

    const double NATURAL_BONUS = 64.0; 

    for (size_t u = 0; u < candidates.size(); u++) {
        for (auto& edge : candidates[u]) {
            double oracle_delta = edge.cost; 
            double segment_hot_cost = segments[edge.to].hot_cost; 
            double segment_pretty_warm_cost = oracle_delta + segment_hot_cost;
            double segment_estimated_hot_cost = segment_pretty_warm_cost / warm_to_hot_ratio;
            edge.cost = segment_estimated_hot_cost - segment_hot_cost;

            if (edge.to == u + 1) {
                total_natural_cost_after_normalizing_should_be_zero += edge.cost;
                edge.cost -= NATURAL_BONUS;
            }
        }
    }
    std::cout << "Total natural cost after normalizing (should be zero): " << total_natural_cost_after_normalizing_should_be_zero << std::endl;
    
    // =========================================================
    // PHASE 3: MICRO-SA (Clustering via CSA Library)
    // =========================================================
    std::cout << "\nStarting Phase 3: Micro-SA (Clustering)..." << std::endl;

    CSA::Solver<int, double> solver;
    solver.m = 8; 
    solver.max_iterations = 4000000;
    solver.tgen_initial = 0.05;
    solver.tgen_schedule = 0.999995;
    solver.tacc_initial = 1000.0;
    solver.tacc_schedule = 0.001; 
    solver.desired_variance = 0.99;

    std::vector<int> initial_x(num_segments);
    for(size_t i=0; i<num_segments; ++i) {
        initial_x[i] = (i < num_segments - 1) ? (int)(i + 1) : -1;
    }

    ProblemData prob_data;
    prob_data.num_segments = num_segments;
    prob_data.candidates = &candidates;

    solver.minimize(
        num_segments,
        initial_x.data(),
        csa_cost_function,
        csa_step_function,
        csa_progress,
        &prob_data
    );

    std::cout << "\nCSA Optimization Complete." << std::endl;

    // =========================================================
    // PHASE 4: GREEDY STITCHER
    // =========================================================
    std::cout << "Starting Phase 4: Greedy Stitcher..." << std::endl;
    
    std::vector<int> next_seg = initial_x; 
    std::vector<int> prev_seg(num_segments, -1);
    
    // Clear prev_seg from any garbage, rebuild strictly from next_seg
    for(size_t i=0; i<num_segments; ++i) {
        if (next_seg[i] != -1) {
            prev_seg[next_seg[i]] = i;
        }
    }

    struct ChainObj {
        int head;
        int tail;
    };
    std::list<ChainObj> chains;
    std::vector<bool> visited(num_segments, false);

    // Identify Linear Chains
    for(size_t i=0; i<num_segments; ++i) {
        if (prev_seg[i] == -1 && !visited[i]) {
            int c = i;
            int safety = 0;
            while(c != -1) { 
                visited[c]=true; 
                if (next_seg[c] == -1) break; // Tail found
                c = next_seg[c]; 
                if (++safety > (int)num_segments * 2) { 
                    // Cycle detected inside a "Head" chain? Should not happen if cost fx worked.
                    // Break it here.
                    next_seg[c] = -1; 
                    break;
                }
            }
            chains.push_back({(int)i, c});
        }
    }
    
    // Break Cycles that don't have heads (Rings)
    for(size_t i=0; i<num_segments; ++i) {
        if (!visited[i]) {
            // Snipping the ring to make a chain
            int prev = prev_seg[i];
            if (prev != -1) { next_seg[prev] = -1; prev_seg[i] = -1; } 
            
            int c = i;
            int safety = 0;
            while(c != -1) { 
                visited[c]=true; 
                if (next_seg[c] == -1) break;
                c = next_seg[c]; 
                if (++safety > (int)num_segments * 2) { next_seg[c] = -1; break; }
            }
            chains.push_back({(int)i, c});
        }
    }

    std::cout << "Initial Fragments: " << chains.size() << std::endl;

    while (chains.size() > 1) {
        double best_merge_cost = 1e18;
        std::list<ChainObj>::iterator best_pred, best_succ;
        bool found = false;

        for (auto it1 = chains.begin(); it1 != chains.end(); ++it1) {
            for (auto it2 = chains.begin(); it2 != chains.end(); ++it2) {
                if (it1 == it2) continue;
                
                int tail = it1->tail;
                int head = it2->head;
                
                double cost = 2000.0;
                bool known = false;
                for(const auto& e : candidates[tail]) {
                    if ((int)e.to == head) { cost = e.cost; known=true; break; }
                }
                
                if (cost < best_merge_cost) {
                    best_merge_cost = cost;
                    best_pred = it1;
                    best_succ = it2;
                    found = true;
                }
            }
        }

        if (found) {
            next_seg[best_pred->tail] = best_succ->head;
            prev_seg[best_succ->head] = best_pred->tail;
            best_pred->tail = best_succ->tail;
            chains.erase(best_succ);
            if (chains.size() % 10 == 0) std::cout << "\rRemaining Chains: " << chains.size() << "   " << std::flush;
        } else {
            break;
        }
    }
    std::cout << "\nFinal Merge Complete." << std::endl;

    // =========================================================
    // 5. EXPORT (FIXED: SAFETY BREAK)
    // =========================================================
    int head = chains.front().head;
    std::ofstream idx_ofs(original_file_base + ".index");
    std::vector<size_t> order;
    
    int curr = head;
    bool first = true;
    int safety_counter = 0;
    
    while(curr != -1) {
        order.push_back(curr);
        if(!first) idx_ofs << " ";
        idx_ofs << curr;
        first = false;
        curr = next_seg[curr];
        
        // SAFETY BREAK: Prevent 6GB file if cycle persists
        if (++safety_counter > (int)num_segments + 100) {
            std::cerr << "\nERROR: Infinite loop detected during export! Terminating export." << std::endl;
            break;
        }
    }
    idx_ofs.close();

    std::ifstream data_ifs(original_file_base, std::ios::binary | std::ios::ate);
    size_t f_size = data_ifs.tellg(); data_ifs.seekg(0);
    std::vector<uint8_t> buf(f_size); data_ifs.read((char*)buf.data(), f_size);
    data_ifs.close();

    std::ofstream rofs(original_file_base + ".reordered", std::ios::binary);
    for (size_t s : order) rofs.write((char*)&buf[segments[s].start], segments[s].length);
    rofs.close();
}