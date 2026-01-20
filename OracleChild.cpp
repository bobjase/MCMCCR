#include <windows.h>
#include <eh.h>
#include <vector>
#include <array>
#include <algorithm>
#include <io.h>
#include <fcntl.h>
#include <cctype>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <fstream>

#include "EntropySegmenter.h"
#include "CCRConfig.h"
#include "Util.hpp"
#include "Archive.hpp"
#include "CM.hpp"
#include "CM-inl.hpp"
#include "DeltaFilter.hpp"
#include "Dict.hpp"
#include "File.hpp"
#include "Huffman.hpp"
#include "LZ-inl.hpp"
#include "ProgressMeter.hpp"
#include "Tests.hpp"
#include "TurboCM.hpp"
#include "X86Binary.hpp"
#include "fingerprint.h"
#include "SegmentFile.h"

struct SegmentInfo {
    size_t start;
    size_t length;
    double hot_cost;
};

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

// --- Oracle Child Process Function ---
int OracleChildMain(int argc, char* argv[]) {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setbuf(stdout, NULL);  // Unbuffered
    debugLog("OracleChildMain start");
    HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD written;
    try {
        if (argc < 4) {
            debugError("Error: argc < 4");
            return 1;
        }
        std::string original_file = argv[2];
        std::string segments_file = argv[3];

        // Load segments from CSV
        std::vector<Segment> segments;
        try {
            segments = readSegments(segments_file);
        } catch (const std::exception& e) {
            debugError("Error reading segments from " + segments_file + ": " + e.what());
            return 1;
        }
        uint64_t num_segments = segments.size();
        debugLog("num_segments: " + std::to_string(num_segments));

        // Convert to SegmentInfo for compatibility
        std::vector<SegmentInfo> valid_segments(num_segments);
        for (size_t i = 0; i < num_segments; ++i) {
            valid_segments[i].start = segments[i].startByte;
            valid_segments[i].length = segments[i].lengthBytes;
            valid_segments[i].hot_cost = segments[i].entropyBits; // Use entropy bits as hot_cost
        }
        debugLog("converted segments to SegmentInfo");

        // Memory map original file
        HANDLE hFile = CreateFileA(original_file.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            debugError("CreateFile failed for " + original_file);
            return 1;
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize)) {
            debugError("GetFileSizeEx failed");
            CloseHandle(hFile);
            return 1;
        }
        size_t file_size = fileSize.QuadPart;

        HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping == NULL) {
            debugError("CreateFileMapping failed");
            CloseHandle(hFile);
            return 1;
        }

        LPVOID pView = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (pView == NULL) {
            debugError("MapViewOfFile failed");
            CloseHandle(hMapping);
            CloseHandle(hFile);
            return 1;
        }
        char* file_data = (char*)pView;

        // Adjust valid_segments for file size
        for (auto& p : valid_segments) {
            size_t start = p.start;
            size_t len = p.length;
            if (start >= file_size || len == 0) continue;
            if (start + len > file_size) {
                len = file_size - start;
                p.length = len;
            }
        }

        // Define CM objects ONCE (performance optimization)
        // [CCR ALIGNMENT] Define CM objects ONCE
        // 1. Joint Compressor (Predecessor + Successor)
        // cm::CM<16, false> cm(FrequencyCounter<256>(), 0, false, Detector::kProfileText);
        
        // // Manual Configuration to match -x11
        // cm.init();
        // cm.text_profile_ = GetCCRProfile(); // Inject Economy Profile
        // cm.SetDataProfile(cm::CM<16, false>::kProfileText); // Force Text Mode
        // cm.skip_init = true; // Prevent accidental reset
        // cm.observer_mode = true;

          // Define CM objects ONCE (performance optimization)
        cm::CM<16, false> cm(FrequencyCounter<256>(), 0, false, Detector::kProfileText);
        // Use full profile for max compression accuracy
        //cm.SetDataProfile(cm::CM<16, false>::kProfileText); // Force Text Mode
        cm.text_profile_ = GetCCRProfile(); // Inject Economy Profile
        //cm.cur_profile_ = cm::CMProfile();
        // for (int i = 0; i < static_cast<int>(cm::kModelCount); ++i) {
        //     cm.cur_profile_.EnableModel(static_cast<cm::ModelType>(i));
        // }
        //cm.cur_profile_.SetMatchModelOrder(12);
        //cm.cur_profile_.SetMinLZPLen(10);
        cm.observer_mode = true;
        cm.init();
        cm.text_profile_ = GetCCRProfile(); // Inject Economy Profile
        size_t max_segment_length = 1024000;


        // --- GLOBAL MEAT PRIMING ---
        debugLog("Priming global meat snapshot...");

        // 1. Wipe and Reset for a fresh start
        memset(cm.base_hash_table_, 0, cm.hash_alloc_size_);
        cm.init();
        cm.entropies.clear();
        cm.SetOverlay(nullptr);

        // 2. Perform the sparse scan (Stride 4)
        VoidWriteStream out_prime; 
        for (size_t i = 0; i < valid_segments.size(); i += std::floor(static_cast<double>(valid_segments.size()) / 12.0)) {
            size_t start = valid_segments[i].start;
            size_t len = valid_segments[i].length;
            
            // Safety check for file data
            const uint8_t* ptr_start = (const uint8_t*)(file_data + start);
            ReadMemoryStream in_prime(ptr_start, ptr_start + len);
            
            // We don't care about the entropy result here, just the hash table updates
            cm.compress(&in_prime, &out_prime, len);
        }

        // 3. Capture the "Meat" Snapshot
        auto global_meat_snapshot = cm.takeSnapshot();
        debugLog("Global meat snapshot captured.");
        // ----------------------------

        // Worker loop
        while (true) {
            // Read pred_id
            size_t pred_id;
            if (fread(&pred_id, sizeof(size_t), 1, stdin) != 1) {
                debugError("Failed to read pred_id " + std::to_string(pred_id)  + ", exiting " );
                break;
            }
            //debugError("read pred_id: " + std::to_string(pred_id));
            if (pred_id >= valid_segments.size()) {
                debugError("Invalid pred_id");
                continue;
            }

            // Read succ_list
            uint64_t num_succ;
            if (fread(&num_succ, sizeof(uint64_t), 1, stdin) != 1) {
                debugError("Failed to read num_succ");
                break;
            }
            debugLog("read num_succ: " + std::to_string(num_succ));
            std::vector<size_t> succ_list(num_succ);
            if (num_succ > 0 && fread(succ_list.data(), sizeof(size_t), num_succ, stdin) != num_succ) {
                debugError("Failed to read succ_list");
                break;
            }
            debugLog("read succ_list");

            // Read alone_costs
            std::vector<double> alone_costs(num_succ);
            if (num_succ > 0 && fread(alone_costs.data(), sizeof(double), num_succ, stdin) != num_succ) {
                debugError("Failed to read alone_costs");
                break;
            }
            debugLog("read alone_costs");

            // Create succ to alone cost map
            std::vector<double> succ_alone_costs(valid_segments.size(), 0.0);
            for (size_t i = 0; i < succ_list.size(); ++i) {
                succ_alone_costs[succ_list[i]] = alone_costs[i];
            }

            // Process the pred
            std::map<size_t, std::vector<std::pair<size_t, double>>> succ_costs;

            // Get pred data
            size_t start_pred = valid_segments[pred_id].start;
            size_t len_pred = valid_segments[pred_id].length;
            std::vector<uint8_t> data_pred(len_pred);
            memcpy(data_pred.data(), file_data + start_pred, len_pred);
            debugLog("Got pred data");

            // --- FIX 1: RUN PREDECESSOR ---
            // We must populate the compressor state with the Predecessor's context
            
            // 1. Clear the Hash Table (Since we disabled the automatic wipe in restoreSnapshot)
            memset(cm.base_hash_table_, 0, cm.hash_alloc_size_);
            
            // 2. Init & Compress
            // cm.init(); 
            // cm.skip_init = true; 
            // cm.entropies.clear(); // <--- CRITICAL FIX: Clear accumulation
            cm.SetOverlay(nullptr); 
            
            ReadMemoryStream in_pred(data_pred.data(), data_pred.data() + data_pred.size());
            VoidWriteStream out_pred; // <--- ADD THIS LINE HERE
            
            cm.compress(&in_pred, &out_pred, data_pred.size());

            // Take pred snapshot for tournament
            auto pred_snapshot = cm.takeSnapshot();
            double pred_cost = 0;//cm.getAccumulatedEntropy();

            // this is always outputting 0 -- seems like a bug
            //debugError("Pred cost: " + std::to_string(pred_cost));

            // 2. TOURNAMENT ROSTER (Fixed Capacity)
            struct Candidate {
                size_t id;
                cm::PagedOverlay* overlay;
                double savings_rate;
            };
            std::vector<Candidate> roster;
            roster.reserve(4096); // Keep it small!

            // The "Working" Overlay (Recycled)
            cm::PagedOverlay* worker_ov = new cm::PagedOverlay(cm.base_hash_table_, cm.hash_alloc_size_);
            
            // Reusable Chunk Buffer
            std::vector<uint8_t> chunk_buf;
            chunk_buf.reserve(4096); 

            // Void output stream for compression
            //VoidWriteStream out_pred;
            debugError("Starting ROUND 1: QUALIFIERS for pred " + std::to_string(pred_id));
            size_t roundOneSize = 256;
            // --- ROUND 1: QUALIFIERS (Streaming & Recycling) ---
            for (size_t succ : succ_list) {
                if (succ >= valid_segments.size()) {
                  debugError("Invalid succ segment size");
                  continue;
                }

                // 1. Setup Chunk
                size_t start = valid_segments[succ].start;
                size_t len = valid_segments[succ].length;
                size_t scan_len = std::min((size_t)roundOneSize, len);

                // 2. Reset Worker (Reuse memory)
                worker_ov->Reset(); 

                // 3. Run Compression (ZERO COPY)
                cm.SetOverlay(worker_ov);
                cm.restoreSnapshot(pred_snapshot);
                cm.byte_index = 0;
                cm.entropies.clear();

                // Point directly to the memory-mapped file
                const uint8_t* ptr_start = (const uint8_t*)(file_data + start);
                ReadMemoryStream in_chunk(ptr_start, ptr_start + scan_len); // Uses Memory.hpp class
                
                //debugError("Starting compression for succ " + std::to_string(succ));

                cm.compress(&in_chunk, &out_pred, scan_len);

                //debugError("Finished compression for succ " + std::to_string(succ));

                // 4. Score
                // double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                
                // // FIX: Normalize Alone Cost to the scan length
                // // We assume entropy is roughly uniform across the segment (Approximation)
                // size_t full_len = std::min(len, (size_t)2048); // The length used by Parent
                // double alone_rate = succ_alone_costs[succ] / (double)full_len; // Bits per byte
                // double alone_cost_scaled = alone_rate * scan_len; // Cost for *this* chunk

                // // Now compare Apples to Apples (512 bytes vs 512 bytes)
                // double savings_rate = (alone_cost_scaled - joint_delta) / scan_len;

                // measure how much entropy we found (smaller=good) -- we're going to keep the smallest content below
                // simplified from above b/c this is just a tournament and we don't really know how this goes, but we DO know that context importance fades
                double savings_rate = cm.getAccumulatedEntropy() - segments[succ].entropyBytes256;

                // 5. King of the Hill Logic
                if (roster.size() < 4096) {
                    // Always accept if we have room
                    roster.push_back({succ, worker_ov, savings_rate});
                    // Allocate a NEW worker for next turn (since we kept the old one)
                    worker_ov = new cm::PagedOverlay(cm.base_hash_table_, cm.hash_alloc_size_);
                } else {
                    // Find worst in roster
                    auto min_it = std::max_element(roster.begin(), roster.end(), 
                      [](const Candidate& a, const Candidate& b){ 
                          if (std::abs(a.savings_rate - b.savings_rate) > 1e-9) return a.savings_rate < b.savings_rate;
                          return a.id > b.id; // Break ties: prefer evicting higher IDs (arbitrary but consistent)
                      });
                    
                    if (savings_rate > min_it->savings_rate) {
                        // We beat the worst! 
                        // Swap overlays: We take the roster spot, and the loser becomes the new worker.
                        cm::PagedOverlay* recycled_ov = min_it->overlay;
                        
                        *min_it = {succ, worker_ov, savings_rate}; // Replace entry
                        
                        worker_ov = recycled_ov; // Recycle the loser
                    } else {
                        // We lost. worker_ov remains the worker (will be Reset next loop).
                    }
                }
                debugLog("Processed succ " + std::to_string(succ) + " for pred " + std::to_string(pred_id) + ", roster size: " + std::to_string(roster.size()));
            }
            
            // Cleanup the extra worker
            delete worker_ov;

            // Sort the finalists
            //std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ return a.savings_rate > b.savings_rate; });
            // Deterministic Sort: Break ties with ID
            std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ 
                if (std::abs(a.savings_rate - b.savings_rate) > 1e-9) return a.savings_rate < b.savings_rate;
                return a.id < b.id; // Tie-breaker
            });

            // --- ROUND 2: SEMI-FINALS (2048 Bytes) ---
            for (auto& cand : roster) {
                size_t len = valid_segments[cand.id].length;
                if (len <= roundOneSize) continue; // Already fully scanned
                
                size_t limit = 2048;
                size_t scan_len = std::min(len, limit);
                size_t start = valid_segments[cand.id].start;
                
                std::vector<uint8_t> chunk(scan_len);
                memcpy(chunk.data(), file_data + start, scan_len);

                // Reuse Overlay (it contains the hash writes from Round 1)
                cm.SetOverlay(cand.overlay);
                cand.overlay -> Reset();
                cm.restoreSnapshot(pred_snapshot); // Reset Mixers to start
                cm.byte_index = 0;
                cm.entropies.clear();
                
                // RESTART Scan (0 -> 2048)
                // Why restart? Because the Overlay preserved the Hash Table, but we reset the Mixers.
                // Re-running the first 256 bytes is negligible cost.
                MemoryReadStream in_chunk(chunk);
                cm.compress(&in_chunk, &out_pred, scan_len);

                //double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                
                // // FIX: Normalize Alone Cost
                // size_t full_len = std::min(len, max_segment_length);
                // double alone_rate = succ_alone_costs[cand.id] / (double)full_len;
                // double alone_cost_scaled = alone_rate * scan_len;
                
                // cand.savings_rate = (alone_cost_scaled - joint_delta) / scan_len;


                // measure how much entropy we found (smaller=good) -- we're going to keep the smallest content below
                cand.savings_rate = cm.getAccumulatedEntropy() - segments[cand.id].entropyBytes2048;
            }

            // Prune: Sort & Keep Top 16
            //std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ return a.savings_rate > b.savings_rate; });
            // Deterministic Sort
            std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ 
                if (std::abs(a.savings_rate - b.savings_rate) > 1e-9) return a.savings_rate < b.savings_rate;
                return a.id < b.id; 
            });
            
            size_t cut_2 = std::min(roster.size(), (size_t)512);
            for (size_t k = cut_2; k < roster.size(); ++k) delete roster[k].overlay;
            roster.resize(cut_2);

            // --- ROUND 3: FINALS (Full Segment) ---
            for (const auto& cand : roster) {
                size_t len = valid_segments[cand.id].length;
                size_t scan_len = len; // Full segment
                
                std::vector<uint8_t> chunk(scan_len);
                memcpy(chunk.data(), file_data + valid_segments[cand.id].start, scan_len);
                cm.SetOverlay(cand.overlay);
                cand.overlay -> Reset();
                cm.restoreSnapshot(pred_snapshot);
                cm.byte_index = 0;
                cm.entropies.clear();
                
                MemoryReadStream in_chunk(chunk);
                cm.compress(&in_chunk, &out_pred, scan_len);

                // double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                // double alone_cost = succ_alone_costs[cand.id];
                
                // // Result: Negative Cost = Savings
                // double final_cost = joint_delta - alone_cost; 
                double presumed_compression_size = (scan_len ) * (1.6 /* bits per byte MCM compression */) ;

                // e.g. total entropy = 100k, hot compression = 90kb = this PRED-SUCC pair cost us 10k (though the number is in bits)
                double final_cost = cm.getAccumulatedEntropy() - valid_segments[cand.id].hot_cost;
                succ_costs[cand.id].emplace_back(pred_id, final_cost);
                
                delete cand.overlay; // Done with this candidate
            }
            cm.entropies.clear(); // <--- CRITICAL FIX: Clear accumulation
            cm.SetOverlay(nullptr); // Safety reset

            debugLog("loop end");
            // Channel separation: stdout for data, stderr for debug
            debugLog("before writing results");
            uint64_t num_results = succ_costs.size();
            WriteFile(hStdout, &num_results, sizeof(uint64_t), &written, NULL);
            FlushFileBuffers(hStdout);
            for (const auto& succ_pair : succ_costs) {
                size_t succ = succ_pair.first;
                const auto& costs = succ_pair.second;
                WriteFile(hStdout, &succ, sizeof(size_t), &written, NULL);
                FlushFileBuffers(hStdout);
                uint64_t num_costs = costs.size();
                WriteFile(hStdout, &num_costs, sizeof(uint64_t), &written, NULL);
                FlushFileBuffers(hStdout);
                for (const auto& cost_pair : costs) {
                    WriteFile(hStdout, &cost_pair.first, sizeof(size_t), &written, NULL);
                    FlushFileBuffers(hStdout);
                    WriteFile(hStdout, &cost_pair.second, sizeof(double), &written, NULL);
                    FlushFileBuffers(hStdout);
                }
            }
            debugLog("after writing results");
        }
        debugLog("OracleChildMain end");
        UnmapViewOfFile(pView);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 0;
    } catch (const std::bad_alloc& e) {
        debugError("Bad alloc: " + std::string(e.what()));
        return 1;
    } catch (const std::out_of_range& e) {
        debugError("Out of range: " + std::string(e.what())); 
        return 1;
    } catch (const std::runtime_error& e) {
        debugError("Runtime error: " + std::string(e.what()));    
        return 1;
    } catch (const std::exception& e) {
        debugError("Exception: " + std::string(e.what()));  
        return 1;
    } catch (...) {
        debugError("Unknown exception caught"); 
        return 1;
    }
}