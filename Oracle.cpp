#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cerrno>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <map>
#include <windows.h>
#include "File.hpp"
#include "CM.hpp"
#include "CM-inl.hpp"
#include "CCRConfig.h"
#include "Util.hpp"
#include "Archive.hpp"
#include "ProgressMeter.hpp"
#include "SegmentFile.h"

class MemoryReadStream : public Stream {
  const std::vector<uint8_t>& data;
  size_t pos;
public:
  MemoryReadStream(const std::vector<uint8_t>& d) : data(d), pos(0) {}
  int get() override {
    if (pos >= data.size()) return EOF;
    return data[pos++];
  }
  void put(int) override {} // no-op
  size_t read(uint8_t* buf, size_t n) override {
    size_t to_read = std::min(n, data.size() - pos);
    if (to_read > 0) {
      memcpy(buf, &data[pos], to_read);
      pos += to_read;
    }
    return to_read;
  }
  void write(const uint8_t*, size_t) override {} // no-op
  void seek(uint64_t p) override { pos = p; }
  uint64_t tell() const override { return pos; }
};

struct SegmentInfo {
    size_t start;
    size_t length;
    double hot_cost;
};

int runOracle(const std::string& in_file) {
    std::cout << "Running Oracle Mode" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "in_file: " << in_file << std::endl;
    std::ifstream ifs(in_file + ".segments.candidates");
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    content.erase(std::remove(content.begin(), content.end(), '\r'), content.end());
    std::vector<std::vector<size_t>> candidates;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line, '\n')) {
      if (line.empty()) continue;
      std::istringstream iss_line(line);
      std::string token;
      std::getline(iss_line, token, ':');
      size_t i;
      try {
          i = std::stoul(token);
      } catch (const std::exception&) {
          std::cerr << "Invalid pred index in candidates: " << token << std::endl;
          continue;
      }
      if (i >= candidates.size()) candidates.resize(i + 1);
      std::vector<size_t>& cands = candidates[i];
      while (std::getline(iss_line, token, ',')) {
        // Trim leading/trailing whitespace
        token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());
        if (!token.empty()) {
          size_t pos = token.find_first_not_of("0123456789");
          if (pos != std::string::npos) token = token.substr(0, pos);
          if (!token.empty()) {
            try {
                cands.push_back(std::stoul(token));
            } catch (const std::exception&) {
                std::cerr << "Invalid succ index in candidates: " << token << std::endl;
            }
          }
        }
      }
    }
    size_t num_segments = candidates.size();
    std::cout << "Read " << num_segments << " segments from candidates" << std::endl;
    if (candidates.size() > 0) std::cout << "candidates[0].size() = " << candidates[0].size() << std::endl;

    // Read segments from CSV
    std::string segments_file = in_file + ".segments.csv";
    std::cout << "segments_file: " << segments_file << std::endl;
    std::vector<Segment> segments;
    try {
        segments = readSegments(segments_file);
        std::cout << "Loaded " << segments.size() << " segments from " << segments_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error reading segments: " << e.what() << std::endl;
        return 1;
    }
    if (segments.size() != num_segments) {
        std::cerr << "Mismatch: candidates have " << num_segments << " segments, CSV has " << segments.size() << std::endl;
        return 1;
    }

    // Load original file
    std::string original_file = in_file;
    std::cout << "original_file: " << original_file << std::endl;
    std::ifstream fin(original_file, std::ios::binary);
    if (!fin) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    fin.seekg(0, std::ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_data(file_size);
    if (file_size > 0) {
        if (!fin.read((char*)&file_data[0], file_size)) {
            std::cerr << "Error reading original file: " << original_file << std::endl;
            return 1;
        }
        if (fin.gcount() != file_size) {
            std::cerr << "Incomplete read of original file: expected " << file_size << ", got " << fin.gcount() << std::endl;
            return 1;
        }
    }
    fin.close();

    // Filter valid segments as in fingerprint
    std::vector<Segment> valid_segments = segments;

    // Build map: pred -> list of succ
    std::map<size_t, std::vector<size_t>> pred_to_succ;
    
    // REVERTED: Use direct candidate list (presumed Pred -> Succs)
    for (size_t pred = 0; pred < candidates.size(); ++pred) {
      if (!candidates[pred].empty()) {
        std::vector<size_t> clean_succs;
        for (size_t succ : candidates[pred]) {
            // Only keep valid segments and prevent self-loops
            if (succ < num_segments && succ != pred) {
                clean_succs.push_back(succ);
            }
        }
        // Deduplicate simple list
        std::sort(clean_succs.begin(), clean_succs.end());
        clean_succs.erase(std::unique(clean_succs.begin(), clean_succs.end()), clean_succs.end());
        
        if (!clean_succs.empty()) {
            pred_to_succ[pred] = clean_succs;
        }
      }
    }
    std::cout << "pred_to_succ size: " << pred_to_succ.size() << std::endl << std::flush;

    // Natural Baseline Fusion: Inject natural predecessors (pred+1)
    for (size_t pred = 0; pred < num_segments - 1; ++pred) {
      size_t natural_succ = pred + 1;
      if (pred_to_succ.find(pred) == pred_to_succ.end()) {
        pred_to_succ[pred] = std::vector<size_t>();
      }
      // Only add if not already present
      if (std::find(pred_to_succ[pred].begin(), pred_to_succ[pred].end(), natural_succ) == pred_to_succ[pred].end()) {
        pred_to_succ[pred].push_back(natural_succ);
      }
    }
    std::cout << "Injected natural predecessors, pred_to_succ size now: " << pred_to_succ.size() << std::endl << std::flush;

    // For each pred, compress pred once, take snapshot, then evaluate all succ that have pred as candidate
    std::vector<std::vector<std::pair<size_t, double>>> pred_costs(num_segments);  // for each pred, list of (succ, cost)
    std::cout << "Number of pred to process: " << pred_to_succ.size() << std::endl << std::flush;

    // 1. Pre-compute Alone Costs
    std::cout << "Pre-computing Alone Costs: 0.0% (0/" << num_segments << " segments)" << std::flush;
    std::vector<double> global_alone_costs(num_segments);
    size_t max_segment_length = 1024000;
    // Use serial for now
    for (size_t i = 0; i < num_segments; ++i) {
        // Get segment data
        size_t start = valid_segments[i].startByte;
        size_t len = valid_segments[i].lengthBytes;
        if (start >= file_data.size() || len == 0) {
            global_alone_costs[i] = 0.0;
            continue;
        }
        len = std::min(len, file_data.size() - start);
        size_t head_len = std::min(max_segment_length, len);
        std::vector<uint8_t> head_data(head_len);
        memcpy(head_data.data(), file_data.data() + start, head_len);

        // Compress alone
        // [CCR ALIGNMENT] Setup Alone Compressor
        // Use <16, false> and Text Profile to match -x11 exactly.
        cm::CM<16, false> cm_alone(FrequencyCounter<256>(), 0, false, Detector::kProfileText);
        
        // One-time Setup
        cm_alone.init();
        cm_alone.text_profile_ = GetCCRProfile(); // Inject Economy Profile
        cm_alone.SetDataProfile(cm::CM<16, false>::kProfileText);
        cm_alone.skip_init = true; // Lock it
        cm_alone.observer_mode = true;
        cm_alone.entropies.clear(); // <--- CRITICAL FIX: Clear accumulation
        MemoryReadStream in_alone(head_data);
        VoidWriteStream out_alone;
        cm_alone.compress(&in_alone, &out_alone, head_data.size());
        double alone_bits = 0.0;
        for (double e : cm_alone.entropies) alone_bits += e;
        global_alone_costs[i] = alone_bits;

        // Update segment
        segments[i].aloneEntropyBits = alone_bits;
        segments[i].aloneEntropyBitsPerByte = alone_bits / segments[i].lengthBytes;
        // Append "alone" to phaseCompleted if not present
        if (segments[i].phaseCompleted.find("alone") == std::string::npos) {
            if (!segments[i].phaseCompleted.empty()) segments[i].phaseCompleted += ",";
            segments[i].phaseCompleted += "alone";
        }

        // Progress
        double percent = 100.0 * (i + 1) / num_segments;
        std::cout << "\rPre-computing Alone Costs: " << std::fixed << std::setprecision(1) << percent << "% (" << (i + 1) << "/" << num_segments << " segments)" << std::flush;
    }
    std::cout << std::endl;
    std::cout << "Pre-computed alone costs for " << num_segments << " segments" << std::endl;

    // Write updated segments back to CSV
    std::string segments_csv_file = in_file + ".segments.csv";
    writeSegments(segments_csv_file, segments);
    std::cout << "Updated segments with alone costs in " << segments_csv_file << std::endl;

    // --- NEW: SAVE .ALONE FILE ---
    std::string alone_out_file = in_file + ".alone";
    std::ofstream alone_ofs(alone_out_file); // Text format for readability/simplicity
    if (alone_ofs) {
        for (double cost : global_alone_costs) {
            alone_ofs << cost << "\n";
        }
        alone_ofs.close();
        std::cout << "Wrote alone costs to " << alone_out_file << std::endl;
    } else {
        std::cerr << "Failed to write .alone file!" << std::endl;
    }
    // -----------------------------

    // 2. Swarm Logic
    std::vector<std::pair<size_t, std::vector<size_t>>> pred_list;
    for (const auto& p : pred_to_succ) {
        pred_list.emplace_back(p.first, p.second);
    }
    size_t total_preds = pred_list.size();
    std::atomic<size_t> global_idx(0);
    std::mutex results_mutex;
    std::atomic<size_t> processed_preds(0);
    std::mutex output_mutex;
    const int BATCH_SIZE = 1;

    auto worker_func = [&]() {
        //std::cout << "Starting worker thread..." << std::endl;
        while (true) {
            size_t idx = global_idx.load();
            if (idx >= total_preds) return;

            // Spawn Process
            char exe_path[MAX_PATH];
            GetModuleFileNameA(NULL, exe_path, MAX_PATH);
            std::string cmd = std::string(exe_path) + " -oracle-child \"" + original_file + "\" \"" + segments_file + "\"";
            STARTUPINFOA si = {sizeof(STARTUPINFOA)};
            si.dwFlags = STARTF_USESTDHANDLES;
            PROCESS_INFORMATION pi;
            HANDLE hChildInRead, hChildInWrite, hChildOutRead, hChildOutWrite;
            SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
            CreatePipe(&hChildInRead, &hChildInWrite, &sa, 0);
            CreatePipe(&hChildOutRead, &hChildOutWrite, &sa, 0);
            si.hStdInput = hChildInRead;
            si.hStdOutput = hChildOutWrite;
            si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
            SetHandleInformation(hChildInWrite, HANDLE_FLAG_INHERIT, 0);
            SetHandleInformation(hChildOutRead, HANDLE_FLAG_INHERIT, 0);
            if (!CreateProcessA(NULL, const_cast<char*>(cmd.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
                std::cerr << "Failed to create child process" << std::endl;
                return;
            }
            // Set priority to IDLE
            if (!SetPriorityClass(pi.hProcess, IDLE_PRIORITY_CLASS)) {
                std::cerr << "SetPriorityClass failed: " << GetLastError() << std::endl;
            }
            CloseHandle(hChildInRead);
            CloseHandle(hChildOutWrite);

            // Process Batch
            int processed_in_batch = 0;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                size_t current_idx = global_idx.fetch_add(1);
                if (current_idx >= total_preds) break;

                // Prepare Data
                size_t pred_id = pred_list[current_idx].first;
                const auto& succs = pred_list[current_idx].second;
                std::vector<double> succ_alone_costs;
                for (size_t s : succs) succ_alone_costs.push_back(global_alone_costs[s]);

                // Write to Child
                DWORD written;
                if (!WriteFile(hChildInWrite, &pred_id, sizeof(size_t), &written, NULL) || written != sizeof(size_t)) {
                    std::cerr << "Failed to write pred_id to child" << std::endl;
                    break;
                }
                uint64_t num_succs = succs.size();
                if (!WriteFile(hChildInWrite, &num_succs, sizeof(uint64_t), &written, NULL) || written != sizeof(uint64_t)) {
                    std::cerr << "Failed to write num_succs to child" << std::endl;
                    break;
                }
                if (!succs.empty()) {
                    if (!WriteFile(hChildInWrite, succs.data(), sizeof(size_t) * succs.size(), &written, NULL) || written != sizeof(size_t) * succs.size()) {
                        std::cerr << "Failed to write succs to child" << std::endl;
                        break;
                    }
                }
                if (!succ_alone_costs.empty()) {
                    if (!WriteFile(hChildInWrite, succ_alone_costs.data(), sizeof(double) * succ_alone_costs.size(), &written, NULL) || written != sizeof(double) * succ_alone_costs.size()) {
                        std::cerr << "Failed to write succ_alone_costs to child" << std::endl;
                        break;
                    }
                }
                FlushFileBuffers(hChildInWrite);

                // Read from Child
                uint64_t num_results;
                if (!ReadFile(hChildOutRead, &num_results, sizeof(uint64_t), &written, NULL) || written != sizeof(uint64_t)) {
                    std::cerr << "Failed to read num_results from child" << std::endl;
                    break;
                }
                for (uint64_t r = 0; r < num_results; ++r) {
                    size_t succ;
                    if (!ReadFile(hChildOutRead, &succ, sizeof(size_t), &written, NULL) || written != sizeof(size_t)) {
                        std::cerr << "Failed to read succ from child" << std::endl;
                        break;
                    }
                    uint64_t num_costs;
                    if (!ReadFile(hChildOutRead, &num_costs, sizeof(uint64_t), &written, NULL) || written != sizeof(uint64_t)) {
                        std::cerr << "Failed to read num_costs from child" << std::endl;
                        break;
                    }
                    for (uint64_t c = 0; c < num_costs; ++c) {
                        size_t p;
                        if (!ReadFile(hChildOutRead, &p, sizeof(size_t), &written, NULL) || written != sizeof(size_t)) {
                            std::cerr << "Failed to read p from child" << std::endl;
                            break;
                        }
                        double cost;
                        if (!ReadFile(hChildOutRead, &cost, sizeof(double), &written, NULL) || written != sizeof(double)) {
                            std::cerr << "Failed to read cost from child" << std::endl;
                            break;
                        }
                        std::lock_guard<std::mutex> lock(results_mutex);
                        pred_costs[p].emplace_back(succ, cost);
                    }
                }
                processed_in_batch++;
            }

            processed_preds.fetch_add(processed_in_batch);
            {
                std::lock_guard<std::mutex> lock(output_mutex);
                double percent = 100.0 * processed_preds.load() / total_preds;
                std::cout << "\rProcessing: " << std::fixed << std::setprecision(1) << percent << "% (" << processed_preds.load() << "/" << total_preds << " preds)" << std::flush;
            }

            // Kill/Close Child
            CloseHandle(hChildInWrite);
            CloseHandle(hChildOutRead);
            WaitForSingleObject(pi.hProcess, INFINITE);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    };

    // Launch Threads
    int num_cpus = std::thread::hardware_concurrency();
    if (num_cpus == 0) num_cpus = 1;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_cpus; ++i) threads.emplace_back(worker_func);
    for (auto& t : threads) t.join();
    std::cout << std::endl;

    std::cout << "Finished processing all pred" << std::endl << std::flush;

    // Output .oracle file
    std::cout << "Starting to write oracle file" << std::endl << std::flush;
    std::string out_file = in_file + ".oracle";
    std::ofstream ofs(out_file);
    if (!ofs) {
      std::cerr << "Error opening output file: " << out_file << std::endl;
      return 1;
    }
    // Output per pred: pred : succ1,cost1 ; succ2,cost2 ; ...
    for (size_t pred = 0; pred < num_segments; ++pred) {
      if (!pred_costs[pred].empty()) {
        ofs << pred << ":";
        for (size_t i = 0; i < pred_costs[pred].size(); ++i) {
          if (i > 0) ofs << ";";
          ofs << pred_costs[pred][i].first << "," << pred_costs[pred][i].second;
        }
        ofs << std::endl;
      }
    }
    ofs.close();
    std::cout << "Wrote oracle results to " << out_file << std::endl << std::flush;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << std::endl;  // Clear progress line
    std::cout << "Oracle processing completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}