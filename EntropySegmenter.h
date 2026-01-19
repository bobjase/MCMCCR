#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

class EntropySegmenter {
public:
    struct Segment {
        size_t start;
        size_t end;
        double mean_entropy;
        double variance;
    };

    // The "Penalty" controls granularity.
    // Higher = Fewer, cleaner segments (Significant changes only).
    // Lower = More segments (Sensitive to small jitters).
    // Recommended: 2 * log(N) (BIC Criterion) or tuned manually (e.g., 10.0 - 50.0).
    static std::vector<size_t> FindCuts(const std::vector<double>& entropy, double penalty, size_t min_segment_size = 512) {
        size_t n = entropy.size();
        if (n < min_segment_size) return {};

        // 1. Precompute Cumulative Sums for O(1) Cost Calculation
        std::vector<double> cumsum(n + 1, 0.0);
        std::vector<double> cumsum_sq(n + 1, 0.0);
        for (size_t i = 0; i < n; ++i) {
            cumsum[i+1] = cumsum[i] + entropy[i];
            cumsum_sq[i+1] = cumsum_sq[i] + (entropy[i] * entropy[i]);
        }

        // Helper: Calculate Cost (Negative Log Likelihood of Gaussian)
        // Effectively: Variance * Length
        auto get_cost = [&](size_t s, size_t t) -> double {
            if (t <= s) return 0.0;
            double len = (double)(t - s);
            double sum = cumsum[t] - cumsum[s];
            double sum_sq = cumsum_sq[t] - cumsum_sq[s];
            // Variance * Length = SumSq - (Sum^2)/Len
            double cost = sum_sq - (sum * sum) / len;
            return cost; // This represents the "error" of modeling this segment as a flat line
        };

        // 2. PELT Initialization
        // F[t] = Min cost to segment data up to t
        std::vector<double> F(n + 1, -penalty); // Offset to make math cleaner
        F[0] = -penalty; 
        
        // backtracking[t] = the best start point for the segment ending at t
        std::vector<int> backtracking(n + 1, -1);
        
        // Active candidates for "start point" s.
        // PELT prunes this list to keep complexity O(N).
        std::vector<size_t> candidates;
        candidates.push_back(0);

        // 3. The Loop (O(N) expected)
        for (size_t t = min_segment_size; t <= n; ++t) {
            double min_val = std::numeric_limits<double>::infinity();
            int best_s = -1;
            
            // Only iterate active candidates (Pruned List)
            for (size_t s : candidates) {
                // Enforce minimum segment size
                if (t - s < min_segment_size) continue;

                double cost = F[s] + get_cost(s, t) + penalty;
                if (cost < min_val) {
                    min_val = cost;
                    best_s = (int)s;
                }
            }
            
            F[t] = min_val;
            backtracking[t] = best_s;

            // 4. Pruning Step (The "P" in PELT)
            // Remove candidates s where F[s] + Cost(s,t) > F[t]
            // This proves s can never be optimal for any future point > t
            std::vector<size_t> next_candidates;
            for (size_t s : candidates) {
                if (t - s < min_segment_size) {
                     next_candidates.push_back(s); // Keep if too close to cut yet
                } else {
                     double cost_projection = F[s] + get_cost(s, t);
                     if (cost_projection <= F[t] + penalty) {
                         next_candidates.push_back(s);
                     }
                }
            }
            // Add current t as a candidate for future segments
            next_candidates.push_back(t);
            candidates = next_candidates;
        }

        // 5. Backtrack to find cuts
        std::vector<size_t> cuts;
        int curr = backtracking[n];
        while (curr > 0) {
            cuts.push_back((size_t)curr);
            curr = backtracking[curr];
        }
        std::sort(cuts.begin(), cuts.end());
        return cuts;
    }
};