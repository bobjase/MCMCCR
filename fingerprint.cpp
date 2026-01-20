#include "fingerprint.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <set>
#include <cmath>
#include <map>
#include <thread>
#include <atomic>
#include "File.hpp"
#include "Util.hpp"
#include "SegmentFile.h"

// --- Helper: Fast Rolling Hash (Cyclic Polynomial) ---
inline uint32_t hash_ngram(const uint8_t* data, size_t len) {
    uint32_t h = 0;
    for (size_t i = 0; i < len; ++i) {
        h = (h << 5) ^ (h >> 27) ^ data[i]; // Fast rotate-XOR mix
    }
    return h;
}

// --- Helper: Fast Walsh-Hadamard Transform (In-Place) ---
void fwht(std::vector<float>& data) {
    size_t n = data.size();
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += h * 2) {
            for (size_t j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

// --- Helper: Euclidean GCD ---
int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// --- Helper: GCD of Gaps ---
int compute_gcd_signature(const std::vector<uint8_t>& data) {
    // Track gaps for the most frequent byte
    size_t counts[256] = {0};
    for (uint8_t b : data) counts[b]++;

    uint8_t top_byte = 0;
    size_t max_count = 0;
    for (int i=0; i<256; ++i) if(counts[i] > max_count) { max_count = counts[i]; top_byte = i; }

    if (max_count < 5) return 1; // Not enough data

    int current_gcd = 0;
    int last_pos = -1;

    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == top_byte) {
            if (last_pos != -1) {
                int gap = i - last_pos;
                if (current_gcd == 0) current_gcd = gap;
                else current_gcd = gcd(current_gcd, gap);
                // Optimization: If GCD decays to 1, stop early
                if (current_gcd == 1) return 1;
            }
            last_pos = i;
        }
    }
    return current_gcd;
}

// --- Feature Extraction ---
Fingerprint compute_fingerprint(const std::vector<uint8_t>& segment) {
    Fingerprint f;

    // 1. MinHash (Vocabulary)
    std::set<uint32_t> hashes;
    const size_t kNgram = 4;
    for (size_t i = 0; i + kNgram <= segment.size(); ++i) {
        uint32_t h = hash_ngram(&segment[i], kNgram);
        if (hashes.size() < 256) hashes.insert(h);
        else if (h < *hashes.rbegin()) {
            hashes.erase(std::prev(hashes.end()));
            hashes.insert(h);
        }
    }
    f.minhashes.assign(hashes.begin(), hashes.end());

    // 2. WHT Spectrum (Structure)
    // Analyze first 4096 bytes (power of 2 required for WHT)
    size_t wht_len = 4096;
    std::vector<float> wht_buf(wht_len, 0.0f);
    size_t limit = std::min(segment.size(), wht_len);
    for (size_t i = 0; i < limit; ++i) wht_buf[i] = (segment[i] > 127) ? 1.0f : -1.0f; // Binarize

    fwht(wht_buf);

    // Bin Energy into Dyadic Bands (Sequency 1, 2..3, 4..7, etc.)
    f.wht_energy.resize(12, 0.0f); // log2(4096) = 12 bands
    for (size_t i = 1; i < wht_len; ++i) {
        int band = 0;
        size_t temp = i;
        while (temp >>= 1) band++; // log2(i)
        if (band < 12) f.wht_energy[band] += std::abs(wht_buf[i]);
    }
    // Normalize Energy
    float total_e = 0.0f;
    for(float e : f.wht_energy) total_e += e;
    if(total_e > 0) for(float& e : f.wht_energy) e /= total_e;

    // 3. GCD (Alignment)
    f.gcd_peaks.push_back(compute_gcd_signature(segment)); // Only primary GCD for speed

    // 4. Volatility (Entropy Proxy)
    // Use Shannon Entropy of Histogram as robust proxy
    float hist[256] = {0};
    for(uint8_t b : segment) hist[b]++;
    float ent = 0;
    float inv_len = 1.0f / segment.size();
    for(int i=0; i<256; ++i) {
        if(hist[i] > 0) {
            float p = hist[i] * inv_len;
            ent -= p * std::log2(p);
        }
    }
    f.volatility = ent;

    return f;
}

// --- Distance Metric ---
DistanceComponents compute_raw_dist(const Fingerprint& donor, const Fingerprint& recv) {
    DistanceComponents dists = {0.0f, std::vector<float>(12, 0.0f), 0.0f, 0.0f};

    // 1. Asymmetric Containment (Vocabulary)
    // Score = |Intersection| / |Recv|
    // "How much of Recv's dictionary does Donor have?"
    size_t matches = 0;
    size_t i=0, j=0;
    while(i < donor.minhashes.size() && j < recv.minhashes.size()) {
        if (donor.minhashes[i] < recv.minhashes[j]) i++;
        else if (donor.minhashes[i] > recv.minhashes[j]) j++;
        else { matches++; i++; j++; }
    }
    float containment = (recv.minhashes.empty()) ? 0.0f : (float)matches / recv.minhashes.size();
    dists.vocab = 1.0f - containment; // 0.0 = Perfect Containment

    // 2. Structural Distance (WHT)
    dists.wht.resize(12);
    for(size_t k=0; k<12; ++k) {
        dists.wht[k] = std::abs(donor.wht_energy[k] - recv.wht_energy[k]);
    }

    // 3. Alignment Mismatch (GCD)
    // Penalty if they have different non-trivial GCDs
    float gcd_penalty = 0.0f;
    int g1 = donor.gcd_peaks[0];
    int g2 = recv.gcd_peaks[0];
    if (g1 > 1 && g2 > 1 && g1 != g2 && (g1 % g2 != 0) && (g2 % g1 != 0)) {
        gcd_penalty = 0.5f; // Strong penalty for mismatched grids
    }
    dists.gcd = gcd_penalty;

    // 4. Volatility Gradient (Cooling Schedule)
    // Penalize: Structured (Low Ent) -> Noisy (High Ent)
    float vol_penalty = 0.0f;
    if (recv.volatility > donor.volatility + 0.5f) {
        vol_penalty = (recv.volatility - donor.volatility) * 0.5f;
    }
    dists.vol = vol_penalty;

    return dists;
}

float hybrid_distance(const Fingerprint& donor, const Fingerprint& recv) {
    DistanceComponents dists = compute_raw_dist(donor, recv);

    // --- Calibrated Weights (enwik8 Deep Dive) ---
    // WHT Weights: Derived from 550k sample regression.
    // Vocab Weight: Manually boosted to 0.25 to ensure content matching 
    //               remains active within the structural safety zones.
    
    float w_vocab = 0.2500f; // Boosted from 0.007
    float w_gcd   = 0.0000f;
    float w_vol   = 0.0000f;
    
    // Band 0 dominates (1.0) to prevent Binary/Text mixing.
    // Band 1 (0.36) detects high-freq texture.
    const float w_wht[] = {
        1.0000f, 0.3643f, 0.0185f, 0.0690f,
        0.0179f, 0.0246f, 0.0177f, 0.0298f,
        0.0195f, 0.0183f, 0.0183f, 0.0002f
    };

    float wht_score = 0.0f;
    for(size_t k=0; k<12; ++k) {
        wht_score += dists.wht[k] * w_wht[k];
    }

    return (dists.vocab * w_vocab) + wht_score + (dists.gcd * w_gcd) + (dists.vol * w_vol);
}

struct SegmentInfo {
    size_t start;
    size_t length;
    double hot_cost;
};

// --- Helper: Binary I/O ---
template <typename T>
void WriteBinary(const std::string& filename, const std::vector<T>& vec) {
    FILE* f = fopen(filename.c_str(), "wb");
    uint64_t size = vec.size();
    fwrite(&size, sizeof(uint64_t), 1, f);
    if (size > 0) fwrite(vec.data(), sizeof(T), size, f);
    fclose(f);
}

static void printHeader() {
    std::cout << "MCM file compressor" << std::endl;
}

void runFingerprint(const std::string& original_file, int top_k) {
    printHeader();
    std::cout << "Running Hybrid Holographic Fingerprinting Mode" << std::endl;

    std::string in_file = original_file;
    std::cout << "in_file: " << in_file << std::endl;

    // Read segments from CSV
    std::string segments_file = in_file + ".segments.csv";
    std::vector<Segment> segments;
    try {
        segments = readSegments(segments_file);
        std::cout << "Loaded " << segments.size() << " segments from " << segments_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error reading segments: " << e.what() << std::endl;
        return;
    }

    // Determine original file
    std::string original_file_path = in_file;

    // Load original file
    File fin;
    if (fin.open(original_file_path, std::ios_base::in | std::ios_base::binary)) {
        std::cerr << "Error opening original file: " << original_file_path << std::endl;
        return;
    }
    std::vector<uint8_t> file_data(fin.length());
    fin.read(&file_data[0], fin.length());
    fin.close();

    // Filter valid segments
    std::vector<std::pair<size_t, size_t>> valid_segments;
    for (auto& seg : segments) {
        size_t start = seg.startByte;
        size_t len = seg.lengthBytes;
        if (start >= file_data.size() || len == 0) continue;
        if (start + len > file_data.size()) {
            len = file_data.size() - start;
        }
        valid_segments.push_back({start, len});
    }
    size_t num_valid = valid_segments.size();

    // Compute Hybrid Fingerprints
    std::vector<Fingerprint> fingerprints(num_valid);
    std::cout << "Computing Hybrid Holographic Fingerprints..." << std::endl;

    // Manual threading using std::thread
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback
    std::vector<std::thread> threads;
    std::atomic<int> progress(0);

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            size_t seg_start = valid_segments[i].first;
            size_t len = valid_segments[i].second;
            size_t scan_len = std::min(len, (size_t)16384);
            std::vector<uint8_t> segment_data(file_data.begin() + seg_start, file_data.begin() + seg_start + scan_len);
            fingerprints[i] = compute_fingerprint(segment_data);
            progress.fetch_add(1, std::memory_order_relaxed);
        }
    };

    int chunk_size = (num_valid + num_threads - 1) / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)num_valid);
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();

    std::cout << "Computed " << progress.load() << " fingerprints." << std::endl;

    // Update segments with fingerprints
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i < fingerprints.size()) {
            // Removed: storing fingerprint string in CSV to reduce bloat
            // std::string fp_str;
            // for (auto h : fingerprints[i].minhashes) {
            //     char buf[9];
            //     sprintf(buf, "%08x", h);
            //     fp_str += std::string(buf) + " ";
            // }
            // if (!fp_str.empty()) fp_str.pop_back(); // remove last space
            // segments[i].fingerprint = fp_str;
            if (!segments[i].phaseCompleted.empty()) segments[i].phaseCompleted += ",";
            segments[i].phaseCompleted += "fingerprint";
        }
    }

    // Matching Loop (Asymmetric)
    std::vector<std::vector<size_t>> candidates(num_valid);

    std::cout << "Matching segments..." << std::endl;
    std::atomic<int> match_progress(0);

    auto match_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            std::vector<std::pair<float, size_t>> distances;
            distances.reserve(num_valid - 1);

            for (size_t j = 0; j < num_valid; ++j) {
                if (i == (int)j) continue;
                float dist = hybrid_distance(fingerprints[j], fingerprints[i]);
                distances.emplace_back(dist, j);
            }

            if (distances.size() > (size_t)top_k) {
                std::partial_sort(distances.begin(), distances.begin() + top_k, distances.end());
                distances.resize(top_k);
            } else {
                std::sort(distances.begin(), distances.end());
            }

            for (const auto& pair : distances) {
                candidates[i].push_back(pair.second);
            }

            match_progress.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> match_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)num_valid);
        match_threads.emplace_back(match_worker, start, end);
    }

    for (auto& th : match_threads) th.join();

    std::cout << "Matched " << match_progress.load() << " segments." << std::endl;

    // Output .candidates file
    std::string out_file = in_file + ".segments.candidates";
    std::ofstream ofs(out_file);
    if (!ofs) {
        std::cerr << "Error opening output file: " << out_file << std::endl;
        return;
    }

    // FIX: Transpose the Graph (Receiver -> Donors  ==>  Donor -> Receivers)
    // The Fingerprinter finds Donors for a Receiver (candidates[recv] = {donor...}).
    // The Oracle expects PRED: SUCC... (candidates[donor] = {recv...}).
    std::vector<std::vector<size_t>> donor_to_receivers(num_valid);
    for (size_t recv = 0; recv < num_valid; ++recv) {
        for (size_t donor : candidates[recv]) {
            if (donor < num_valid) {
                donor_to_receivers[donor].push_back(recv);
            }
        }
    }

    // Write the transposed map (Pred -> Succs)
    for (size_t i = 0; i < num_valid; ++i) {
        ofs << i << ":";
        for (size_t j = 0; j < donor_to_receivers[i].size(); ++j) {
            if (j > 0) ofs << ",";
            ofs << donor_to_receivers[i][j];
        }
        ofs << std::endl;
    }
    ofs.close();
    std::cout << "Wrote transposed candidate lists (Pred->Succs) for " << num_valid << " segments to " << out_file << std::endl;

    // Save fingerprint data for reuse in oracle mode
    std::string fingerprint_file = in_file + ".segments.fingerprints";
    WriteBinary(fingerprint_file, fingerprints);
    std::cout << "Saved " << num_valid << " fingerprint structures to " << fingerprint_file << std::endl;

    // Write updated segments back to CSV
    try {
        writeSegments(segments_file, segments);
        std::cout << "Updated segments with fingerprints in " << segments_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error writing segments: " << e.what() << std::endl;
    }
}