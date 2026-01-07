/*	MCM file compressor

  Copyright (C) 2013, Google Inc.
  Authors: Mathieu Chartier

  LICENSE

    This file is part of the MCM file compressor.

    MCM is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MCM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MCM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <chrono>
#include <ctime>
#include <fstream>
#include <windows.h>
#include <deque>
#include <vector>
#include <io.h>
#include <fcntl.h>

// --- Helper: Binary I/O ---
template <typename T>
void WriteBinary(const std::string& filename, const std::vector<T>& vec) {
    FILE* f = fopen(filename.c_str(), "wb");
    uint64_t size = vec.size();
    fwrite(&size, sizeof(uint64_t), 1, f);
    if (size > 0) fwrite(vec.data(), sizeof(T), size, f);
    fclose(f);
}

template <typename T>
std::vector<T> ReadBinary(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    uint64_t size = 0;
    fread(&size, sizeof(uint64_t), 1, f);
    std::vector<T> vec(size);
    if (size > 0) fread(vec.data(), sizeof(T), size, f);
    fclose(f);
    return vec;
}
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <thread>

#include <omp.h>

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

static constexpr bool kReleaseBuild = false;

struct MarkovMatrix {
  float transitions[8][8];
};

// Byte class mapping: 0-Null/Control, 1-Whitespace, 2-Digits, 3-Lowercase, 4-Uppercase, 5-Punctuation, 6-High-Bit, 7-Other
uint8_t get_byte_class(uint8_t byte) {
  if (byte <= 0x1F || byte == 0x7F) return 0; // Null/Control
  if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r') return 1; // Whitespace
  if (byte >= '0' && byte <= '9') return 2; // Digits
  if (byte >= 'a' && byte <= 'z') return 3; // Lowercase
  if (byte >= 'A' && byte <= 'Z') return 4; // Uppercase
  if ((byte >= 0x21 && byte <= 0x2F) || (byte >= 0x3A && byte <= 0x40) || (byte >= 0x5B && byte <= 0x60) || (byte >= 0x7B && byte <= 0x7E)) return 5; // Punctuation/Symbols
  if (byte >= 0x80) return 6; // High-Bit
  return 7; // Other
}

MarkovMatrix compute_matrix(const std::vector<uint8_t>& segment) {
  MarkovMatrix m = {};
  if (segment.empty()) return m;
  uint8_t prev_class = get_byte_class(segment[0]);
  for (size_t i = 1; i < segment.size(); ++i) {
    uint8_t curr_class = get_byte_class(segment[i]);
    m.transitions[prev_class][curr_class] += 1.0f;
    prev_class = curr_class;
  }
  // Normalize
  for (int i = 0; i < 8; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += m.transitions[i][j];
    if (sum > 0.0f) {
      for (int j = 0; j < 8; ++j) m.transitions[i][j] /= sum;
    }
  }
  return m;
}

float hellinger_distance(const MarkovMatrix& m1, const MarkovMatrix& m2) {
  float dist = 0.0f;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float diff = sqrtf(m1.transitions[i][j]) - sqrtf(m2.transitions[i][j]);
      dist += diff * diff;
    }
  }
  return sqrtf(dist);
}

static void printHeader() {
  std::cout
    << "======================================================================" << std::endl
    << "mcm compressor v" << Archive::Header::kCurMajorVersion << "." << Archive::Header::kCurMinorVersion
    << ", by Mathieu Chartier (c)2016 Google Inc." << std::endl
    << "Experimental, may contain bugs. Contact mathieu.a.chartier@gmail.com" << std::endl
    << "Special thanks to: Matt Mahoney, Stephan Busch, Christopher Mattern." << std::endl
    << "======================================================================" << std::endl;
}

class Options {
public:
  // Block size of 0 -> file size / #threads.
  static const uint64_t kDefaultBlockSize = 0;
  enum Mode {
    kModeUnknown,
    // Compress -> Decompress -> Verify.
    // (File or directory).
    kModeTest,
    // Compress infinite times with different opt vars.
    kModeOpt,
    // In memory test.
    kModeMemTest,
    // Single file test.
    kModeSingleTest,
    // Add a single file.
    kModeAdd,
    kModeExtract,
    kModeExtractAll,
    // Single hand mode.
    kModeCompress,
    kModeDecompress,
    // List & other
    kModeList,
    // Observer mode for entropy profiling
    kModeObserver,
    // Segment mode for entropy-based segmentation
    kModeSegment,
    // Fingerprint mode for structural fingerprinting
    kModeFingerprint,
    // Oracle mode for testing candidate pairs compression
    kModeOracle,
    // Oracle child mode for multiprocess oracle
    kModeOracleChild,
    // PathCover mode for graph construction and reordering
    kModePathCover,
  };
  Mode mode = kModeUnknown;
  bool opt_mode = false;
  CompressionOptions options_;
  Compressor* compressor = nullptr;
  uint32_t threads = 1;
  uint64_t block_size = kDefaultBlockSize;
  FileInfo archive_file;
  std::vector<FileInfo> files;
  const std::string kDictArg = "-dict=";
  const std::string kOutDictArg = "-out-dict=";
  std::string dict_file;
  // Segmentation parameters
  size_t segment_window = 512;
  float segment_threshold = 3.0f;
  size_t segment_min_segment = 2048;
  size_t segment_lookback = 1024;  // Fingerprinting parameters
  size_t fingerprint_top_k = 32;
  int usage(const std::string& name) {
    printHeader();
    std::cout
      << "Caution: Experimental, use only for testing!" << std::endl
      << "Usage: " << name << " [commands] [options] <infile|dir> <outfile>(default infile.mcm)" << std::endl
      << "Options: d for decompress" << std::endl
      << "-{t|f|m|h|x}{1 .. 11} compression option" << std::endl
      << "t is turbo, f is fast, m is mid, h is high, x is max (default " << CompressionOptions::kDefaultLevel << ")" << std::endl
      << "0 .. 11 specifies memory with 32mb .. 5gb per thread (default " << CompressionOptions::kDefaultMemUsage << ")" << std::endl
      << "10 and 11 are only supported on 64 bits" << std::endl
      << "-test tests the file after compression is done" << std::endl
      << "-observer generates entropy profile instead of compressing" << std::endl
      << "-segment performs entropy-based segmentation on .entropy file" << std::endl
      << "-fingerprint performs structural fingerprinting on .segments file" << std::endl
      << "-window <size> smoothing window size (default 64)" << std::endl
      << "-threshold <float> gradient threshold for boundaries (default 0.01)" << std::endl
      << "-min-segment <size> minimum segment size (default 1024)" << std::endl
      << "-lookback <size> lookback distance for inflection points (default 512)" << std::endl
      // << "-b <mb> specifies block size in MB" << std::endl
      // << "-t <threads> the number of threads to use (decompression requires the same number of threads" << std::endl
      << "Examples:" << std::endl
      << "Compress: " << name << " -m9 enwik8 enwik8.mcm" << std::endl
      << "Decompress: " << name << " d enwik8.mcm enwik8.ref" << std::endl;
    return 0;
  }

  int parse(int argc, char* argv[]) {
    assert(argc >= 1);
    std::string program(trimExt(argv[0]));
    // Parse options.
    int i = 1;
    bool has_comp_args = false;
    for (;i < argc;++i) {
      const std::string arg(argv[i]);
      Mode parsed_mode = kModeUnknown;
      if (arg == "-test") parsed_mode = kModeSingleTest; // kModeTest;
      else if (arg == "-memtest") parsed_mode = kModeMemTest;
      else if (arg == "-opt") parsed_mode = kModeOpt;
      else if (arg == "-stest") parsed_mode = kModeSingleTest;
      else if (arg == "-observer") parsed_mode = kModeObserver;
      else if (arg == "-segment") parsed_mode = kModeSegment;
      else if (arg == "-fingerprint") parsed_mode = kModeFingerprint;
      else if (arg == "-oracle") parsed_mode = kModeOracle;
      else if (arg == "-oracle-child") parsed_mode = kModeOracleChild;
      else if (arg == "-pathcover") parsed_mode = kModePathCover;
      else if (arg == "c") parsed_mode = kModeCompress;
      else if (arg == "l") parsed_mode = kModeList;
      else if (arg == "d") parsed_mode = kModeDecompress;
      else if (arg == "a") parsed_mode = kModeAdd;
      else if (arg == "e") parsed_mode = kModeExtract;
      else if (arg == "x") parsed_mode = kModeExtractAll;
      if (parsed_mode != kModeUnknown) {
        if (mode != kModeUnknown) {
          std::cerr << "Multiple commands specified" << std::endl;
          return 2;
        }
        mode = parsed_mode;
        switch (mode) {
          case kModeAdd:
          case kModeExtract:
          case kModeExtractAll: {
            if (++i >= argc) {
              std::cerr << "Expected archive" << std::endl;
              return 3;
            }
            // Archive is next.
            archive_file = FileInfo(argv[i]);
            break;
          }
        }
      } else if (arg == "-opt") opt_mode = true;
      else if (arg == "-filter=none") options_.filter_type_ = kFilterTypeNone;
      else if (arg == "-filter=dict") options_.filter_type_ = kFilterTypeDict;
      else if (arg == "-filter=x86") options_.filter_type_ = kFilterTypeX86;
      else if (arg == "-filter=auto") options_.filter_type_ = kFilterTypeAuto;
      else if (arg.substr(0, std::min(kDictArg.length(), arg.length())) == kDictArg) {
        options_.dict_file_ = arg.substr(kDictArg.length());
      } else if (arg.substr(0, std::min(kOutDictArg.length(), arg.length())) == kOutDictArg) {
        options_.out_dict_file_ = arg.substr(kOutDictArg.length());
      } else if (arg == "-lzp=auto") options_.lzp_type_ = kLZPTypeAuto;
      else if (arg == "-lzp=true") options_.lzp_type_ = kLZPTypeEnable;
      else if (arg == "-lzp=false") options_.lzp_type_ = kLZPTypeDisable;
      else if (arg == "-window") {
        if (i + 1 >= argc) return usage(program);
        segment_window = std::stoull(argv[++i]);
      } else if (arg == "-threshold") {
        if (i + 1 >= argc) return usage(program);
        segment_threshold = std::stod(argv[++i]);
      } else if (arg == "-min-segment") {
        if (i + 1 >= argc) return usage(program);
        segment_min_segment = std::stoull(argv[++i]);
      } else if (arg == "-lookback") {
        if (i + 1 >= argc) return usage(program);
        segment_lookback = std::stoull(argv[++i]);
      } else if (arg == "-b") {
        if (i + 1 >= argc) {
          return usage(program);
        }
        std::istringstream iss(argv[++i]);
        iss >> block_size;
        block_size *= MB;
        if (!iss.good()) {
          return usage(program);
        }
      } else if (arg == "-store") {
        options_.comp_level_ = kCompLevelStore;
        has_comp_args = true;
      } else if (arg[0] == '-') {
        if (arg[1] == 't') options_.comp_level_ = kCompLevelTurbo;
        else if (arg[1] == 'f') options_.comp_level_ = kCompLevelFast;
        else if (arg[1] == 'm') options_.comp_level_ = kCompLevelMid;
        else if (arg[1] == 'h') options_.comp_level_ = kCompLevelHigh;
        else if (arg[1] == 'x') options_.comp_level_ = kCompLevelMax;
        else if (arg[1] == 's') options_.comp_level_ = kCompLevelSimple;
        else {
          std::cerr << "Unknown option " << arg << std::endl;
          return 4;
        }
        has_comp_args = true;
        const std::string mem_string = arg.substr(2);
        if (mem_string == "0") options_.mem_usage_ = 0;
        else if (mem_string == "1") options_.mem_usage_ = 1;
        else if (mem_string == "2") options_.mem_usage_ = 2;
        else if (mem_string == "3") options_.mem_usage_ = 3;
        else if (mem_string == "4") options_.mem_usage_ = 4;
        else if (mem_string == "5") options_.mem_usage_ = 5;
        else if (mem_string == "6") options_.mem_usage_ = 6;
        else if (mem_string == "7") options_.mem_usage_ = 7;
        else if (mem_string == "8") options_.mem_usage_ = 8;
        else if (mem_string == "9") options_.mem_usage_ = 9;
        else if (mem_string == "10" || mem_string == "11") {
          if (sizeof(void*) < 8) {
            std::cerr << arg << " is only supported with 64 bit" << std::endl;
            return usage(program);
          }
          options_.mem_usage_ = (mem_string == "10") ? 10 : 11;
        } else if (!mem_string.empty()) {
          std::cerr << "Unknown mem level " << mem_string << std::endl;
          return 4;
        }
      } else if (!arg.empty()) {
        if (mode == kModeAdd || mode == kModeExtract) {
          files.push_back(FileInfo(argv[i]));  // Read in files.
        } else {
          break;  // Done parsing.
        }
      }
    }
    if (mode == kModeUnknown) {
      const int remaining_args = argc - i;
      // No args, need to figure out what to do:
      // decompress: mcm <archive> 
      // TODO add files to archive: mcm <archive> <files>
      // create archive: mcm <files>
      mode = kModeCompress;
      if (!has_comp_args && remaining_args == 1) {
        // Try to open file.
        File fin;
        if (fin.open(argv[i], std::ios_base::in | std::ios_base::binary) == 0) {
          Archive archive(&fin);
          const auto& header = archive.getHeader();
          if (header.isArchive()) {
            mode = kModeDecompress;
          }
        }
      }
    }
    const bool single_file_mode =
      mode == kModeCompress || mode == kModeDecompress || mode == kModeSingleTest ||
      mode == kModeMemTest || mode == kModeOpt || kModeList || mode == kModeObserver || mode == kModeSegment;
    if (single_file_mode && i < argc) {
      std::string in_file, out_file;
      // Read in file and outfile.
      in_file = argv[i++];
      if (i < argc) {
        out_file = argv[i++];
      } else {
        if (mode == kModeDecompress) {
          out_file = in_file + ".decomp";
        } else {
          out_file = trimDir(in_file) + ".mcm";
        }
      }
      if (mode == kModeObserver) {
        out_file = trimDir(in_file) + ".entropy";
      }
      if (mode == kModeMemTest) {
        // No out file for memtest.
        files.push_back(FileInfo(trimDir(in_file)));
      } else if (mode == kModeCompress || mode == kModeSingleTest || mode == kModeOpt || mode == kModeObserver || mode == kModeSegment) {
        archive_file = FileInfo(trimDir(out_file));
        files.push_back(FileInfo(trimDir(in_file)));
      } else {
        archive_file = FileInfo(trimDir(in_file));
        files.push_back(FileInfo(trimDir(out_file)));
      }
    }
    if (mode != kModeMemTest && mode != kModeOracleChild && mode != kModeOracle &&
      (archive_file.getName().empty() || (files.empty() && mode != kModeList))) {
      std::cerr << "Error, input or output files missing" << std::endl;
      usage(program);
      return 5;
    }
    return 0;
  }
};

extern void RunBenchmarks();

// --- Oracle Child Process Function ---
int OracleChildMain() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    try {
        // Read file_data from stdin
        uint64_t file_size;
        if (fread(&file_size, sizeof(uint64_t), 1, stdin) != 1) {
            std::cerr << "Failed to read file_size" << std::endl;
            return 1;
        }
        std::vector<uint8_t> file_data(file_size);
        if (file_size > 0 && fread(file_data.data(), sizeof(uint8_t), file_size, stdin) != file_size) {
            std::cerr << "Failed to read file_data" << std::endl;
            return 1;
        }

        // Read valid_segments
        uint64_t num_segments;
        if (fread(&num_segments, sizeof(uint64_t), 1, stdin) != 1) {
            std::cerr << "Failed to read num_segments" << std::endl;
            return 1;
        }
        std::vector<std::pair<size_t, size_t>> valid_segments(num_segments);
        for (auto& seg : valid_segments) {
            if (fread(&seg.first, sizeof(size_t), 1, stdin) != 1 ||
                fread(&seg.second, sizeof(size_t), 1, stdin) != 1) {
                std::cerr << "Failed to read segment" << std::endl;
                return 1;
            }
        }

        // Read pred_id
        size_t pred_id;
        if (fread(&pred_id, sizeof(size_t), 1, stdin) != 1) {
            std::cerr << "Failed to read pred_id" << std::endl;
            return 1;
        }

        // Read succ_list
        uint64_t num_succ;
        if (fread(&num_succ, sizeof(uint64_t), 1, stdin) != 1) {
            std::cerr << "Failed to read num_succ" << std::endl;
            return 1;
        }
        std::vector<size_t> succ_list(num_succ);
        if (num_succ > 0 && fread(succ_list.data(), sizeof(size_t), num_succ, stdin) != num_succ) {
            std::cerr << "Failed to read succ_list" << std::endl;
            return 1;
        }

        // Process the pred
        std::map<size_t, std::vector<std::pair<size_t, double>>> succ_costs;

        // Get pred data
        size_t start_pred = valid_segments[pred_id].first;
        size_t len_pred = valid_segments[pred_id].second;
        std::vector<uint8_t> data_pred(file_data.begin() + start_pred, file_data.begin() + start_pred + len_pred);

        // For each succ
        for (size_t succ : succ_list) {
            // Create CM
            cm::CM<6, false> cm(FrequencyCounter<256>(), 4, true, Detector::kProfileText);
            cm.observer_mode = true;

            // Compress full pred
            MemoryReadStream in_pred(data_pred);
            VoidWriteStream out_pred;
            cm.compress(&in_pred, &out_pred, data_pred.size());

            // Get succ data
            size_t start_succ = valid_segments[succ].first;
            size_t len_succ = valid_segments[succ].second;
            std::vector<uint8_t> data_succ(file_data.begin() + start_succ, file_data.begin() + start_succ + len_succ);
            size_t head_len = std::min((size_t)10240, data_succ.size());
            std::vector<uint8_t> head_succ(data_succ.begin(), data_succ.begin() + head_len);

            // Record entropy start
            size_t entropy_start = cm.entropies.size();

            // Compress head
            MemoryReadStream in_head(head_succ);
            VoidWriteStream out_head;
            cm.compress(&in_head, &out_head, head_succ.size());

            // Measure cost
            double transition_cost = 0.0;
            for (size_t k = entropy_start; k < cm.entropies.size(); ++k) {
                transition_cost += cm.entropies[k];
            }

            succ_costs[succ].emplace_back(pred_id, transition_cost);
        }

        // Write results to stdout
        // First, number of succ with costs
        uint64_t num_results = succ_costs.size();
        fwrite(&num_results, sizeof(uint64_t), 1, stdout);
        for (const auto& succ_pair : succ_costs) {
            size_t succ = succ_pair.first;
            const auto& costs = succ_pair.second;
            fwrite(&succ, sizeof(size_t), 1, stdout);
            uint64_t num_costs = costs.size();
            fwrite(&num_costs, sizeof(uint64_t), 1, stdout);
            for (const auto& cost_pair : costs) {
                fwrite(&cost_pair.first, sizeof(size_t), 1, stdout);
                fwrite(&cost_pair.second, sizeof(double), 1, stdout);
            }
        }
        fflush(stdout);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "OracleChildMain exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "OracleChildMain unknown exception" << std::endl;
        return 1;
    }
}

int run_oracle_multiprocess(const char* exe_path, const std::vector<uint8_t>& file_data, const std::vector<std::pair<size_t, size_t>>& valid_segments, const std::map<size_t, std::vector<size_t>>& pred_to_succ, std::vector<std::vector<std::pair<size_t, double>>>& pred_costs) {
    // Detect number of CPUs
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    size_t num_cpus = sysinfo.dwNumberOfProcessors;
    const size_t window_size = std::max<size_t>(1, num_cpus - 2);  // Use up to numCPUs-2, min 1
    std::cout << "Detected " << num_cpus << " CPUs, using window size " << window_size << std::endl;
    std::vector<std::pair<size_t, std::vector<size_t>>> pred_list(pred_to_succ.begin(), pred_to_succ.end());
    std::vector<HANDLE> processes;
    std::vector<HANDLE> child_stdout_reads;
    size_t pred_index = 0;

    while (pred_index < pred_list.size() || !processes.empty()) {
        // Launch new processes up to window size
        while (processes.size() < window_size && pred_index < pred_list.size()) {
            const auto& pair = pred_list[pred_index];
            size_t pred = pair.first;
            const std::vector<size_t>& succ_list = pair.second;

            // Create pipes for child's stdin
            HANDLE hChildStdinRead, hChildStdinWrite;
            SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
            if (!CreatePipe(&hChildStdinRead, &hChildStdinWrite, &sa, 0)) {
                std::cerr << "Failed to create stdin pipe for pred " << pred << std::endl;
                return 1;
            }
            SetHandleInformation(hChildStdinRead, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);  // Child reads
            SetHandleInformation(hChildStdinWrite, HANDLE_FLAG_INHERIT, 0);  // Parent writes, don't inherit

            // Create pipes for child's stdout
            HANDLE hChildStdoutRead, hChildStdoutWrite;
            if (!CreatePipe(&hChildStdoutRead, &hChildStdoutWrite, &sa, 0)) {
                std::cerr << "Failed to create stdout pipe for pred " << pred << std::endl;
                CloseHandle(hChildStdinRead);
                CloseHandle(hChildStdinWrite);
                return 1;
            }
            SetHandleInformation(hChildStdoutRead, HANDLE_FLAG_INHERIT, 0);  // Parent reads, don't inherit
            SetHandleInformation(hChildStdoutWrite, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);  // Child writes

            // Create process
            STARTUPINFO si = {sizeof(STARTUPINFO)};
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = hChildStdinRead;
            si.hStdOutput = hChildStdoutWrite;
            si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

            PROCESS_INFORMATION pi;
            std::string cmd = "mcm.exe -oracle-child";
            if (!CreateProcess(NULL, const_cast<char*>(cmd.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
                std::cerr << "Failed to create process for pred " << pred << std::endl;
                CloseHandle(hChildStdinRead);
                CloseHandle(hChildStdinWrite);
                CloseHandle(hChildStdoutRead);
                CloseHandle(hChildStdoutWrite);
                return 1;
            }

            // Set priority to IDLE
            SetPriorityClass(pi.hProcess, IDLE_PRIORITY_CLASS);

            // Close handles not needed in parent
            CloseHandle(hChildStdinRead);
            CloseHandle(hChildStdoutWrite);

            // Write data to child's stdin
            DWORD written;
            uint64_t file_size = file_data.size();
            WriteFile(hChildStdinWrite, &file_size, sizeof(uint64_t), &written, NULL);
            WriteFile(hChildStdinWrite, file_data.data(), file_data.size(), &written, NULL);

            uint64_t num_seg = valid_segments.size();
            WriteFile(hChildStdinWrite, &num_seg, sizeof(uint64_t), &written, NULL);
            for (const auto& seg : valid_segments) {
                WriteFile(hChildStdinWrite, &seg.first, sizeof(size_t), &written, NULL);
                WriteFile(hChildStdinWrite, &seg.second, sizeof(size_t), &written, NULL);
            }

            WriteFile(hChildStdinWrite, &pred, sizeof(size_t), &written, NULL);

            uint64_t num_succ = succ_list.size();
            WriteFile(hChildStdinWrite, &num_succ, sizeof(uint64_t), &written, NULL);
            if (num_succ > 0) WriteFile(hChildStdinWrite, succ_list.data(), num_succ * sizeof(size_t), &written, NULL);

            // Close stdin write handle
            CloseHandle(hChildStdinWrite);

            processes.push_back(pi.hProcess);
            child_stdout_reads.push_back(hChildStdoutRead);

            std::cout << "Launched process for pred = " << pred << std::endl;
            ++pred_index;
        }

        // Wait for one process to complete
        if (!processes.empty()) {
            DWORD wait_result = WaitForMultipleObjects(processes.size(), processes.data(), FALSE, INFINITE);
            if (wait_result >= WAIT_OBJECT_0 && wait_result < WAIT_OBJECT_0 + processes.size()) {
                size_t completed_index = wait_result - WAIT_OBJECT_0;
                HANDLE completed_process = processes[completed_index];
                HANDLE completed_stdout = child_stdout_reads[completed_index];

                // Get exit code
                DWORD exit_code;
                GetExitCodeProcess(completed_process, &exit_code);
                if (exit_code != 0) {
                    std::cerr << "Child process failed with exit code " << exit_code << std::endl;
                    return 1;
                }

                // Read results from stdout
                uint64_t num_results;
                DWORD read_bytes;
                ReadFile(completed_stdout, &num_results, sizeof(uint64_t), &read_bytes, NULL);
                for (uint64_t i = 0; i < num_results; ++i) {
                    size_t succ;
                    ReadFile(completed_stdout, &succ, sizeof(size_t), &read_bytes, NULL);
                    uint64_t num_costs;
                    ReadFile(completed_stdout, &num_costs, sizeof(uint64_t), &read_bytes, NULL);
                    for (uint64_t j = 0; j < num_costs; ++j) {
                        size_t pred_read;
                        double cost;
                        ReadFile(completed_stdout, &pred_read, sizeof(size_t), &read_bytes, NULL);
                        ReadFile(completed_stdout, &cost, sizeof(double), &read_bytes, NULL);
                        pred_costs[pred_read].emplace_back(succ, cost);
                    }
                }

                // Close handles
                CloseHandle(completed_process);
                CloseHandle(completed_stdout);

                // Remove from vectors
                processes.erase(processes.begin() + completed_index);
                child_stdout_reads.erase(child_stdout_reads.begin() + completed_index);
            }
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
  if (!kReleaseBuild) {
    RunAllTests();
  }
  Options options;
  auto ret = options.parse(argc, argv);
  if (ret) {
    std::cerr << "Failed to parse arguments" << std::endl;
    return ret;
  }
  switch (options.mode) {
  case Options::kModeOracleChild:
    return OracleChildMain();
  case Options::kModeMemTest: {
    constexpr size_t kCompIterations = kIsDebugBuild ? 1 : 1;
    constexpr size_t kDecompIterations = kIsDebugBuild ? 1 : 25;
    // Read in the whole file.
    std::vector<uint64_t> lengths;
    uint64_t long_length = 0;
    for (const auto& file : options.files) {
      File f(file.getName());
      lengths.push_back(f.length());
      long_length += lengths.back();
    }
    auto length = static_cast<size_t>(long_length);
    check(length < 300 * MB);
    auto* in_buffer = new uint8_t[length];
    // Read in the files.
    uint32_t index = 0;
    uint64_t read_pos = 0;
    for (const auto& file : options.files) {
      File f(file.getName(), std::ios_base::in | std::ios_base::binary);
      size_t count = f.read(in_buffer + read_pos, static_cast<size_t>(lengths[index]));
      check(count == lengths[index]);
      index++;
    }
    // Create the memory compressor.
    typedef SimpleEncoder<8, 16> Encoder;
    // auto* compressor = new LZ4;
    // auto* compressor = new LZSSE;
    // auto* compressor = new MemCopyCompressor;
    auto* compressor = new LZ16<FastMatchFinder<MemoryMatchFinder>>;
    auto out_buffer = new uint8_t[compressor->getMaxExpansion(length)];
    uint32_t comp_start = clock();
    uint32_t comp_size;
    static const bool opt_mode = false;
    if (opt_mode) {
      uint32_t best_size = 0xFFFFFFFF;
      uint32_t best_opt = 0;
      std::ofstream opt_file("opt_result.txt");
      for (uint32_t opt = 0; ; ++opt) {
        compressor->setOpt(opt);
        comp_size = compressor->compress(in_buffer, out_buffer, length);
        opt_file << "opt " << opt << " = " << comp_size << std::endl << std::flush;
        std::cout << "Opt " << opt << " / " << best_opt << " =  " << comp_size << "/" << best_size << std::endl;
        if (comp_size < best_size) {
          best_opt = opt;
          best_size = comp_size;
        }
      }
    } else {
      for (uint32_t i = 0; i < kCompIterations; ++i) {
        comp_size = compressor->compress(in_buffer, out_buffer, length);
      }
    }

    const uint32_t comp_end = clock();
    std::cout << "Done compressing " << length << " -> " << comp_size << " = " << float(double(length) / double(comp_size)) << " rate: "
      << prettySize(static_cast<uint64_t>(long_length * kCompIterations / clockToSeconds(comp_end - comp_start))) << "/s" << std::endl;
    memset(in_buffer, 0, length);
    const uint32_t decomp_start = clock();
    for (uint32_t i = 0; i < kDecompIterations; ++i) {
      compressor->decompress(out_buffer, in_buffer, length);
    }
    const uint32_t decomp_end = clock();
    std::cout << "Decompression took: " << decomp_end - comp_end << " rate: "
      << prettySize(static_cast<uint64_t>(long_length * kDecompIterations / clockToSeconds(decomp_end - decomp_start))) << "/s" << std::endl;
    index = 0;
    for (const auto& file : options.files) {
      File f(file.getName(), std::ios_base::in | std::ios_base::binary);
      const auto count = static_cast<uint32_t>(f.read(out_buffer, static_cast<uint32_t>(lengths[index])));
      check(count == lengths[index]);
      for (uint32_t i = 0; i < count; ++i) {
        if (out_buffer[i] != in_buffer[i]) {
          std::cerr << "File" << file.getName() << " doesn't match at byte " << i << std::endl;
          check(false);
        }
      }
      index++;
    }
    std::cout << "Decompression verified" << std::endl;
    break;
  }
  case Options::kModeSingleTest:
  case Options::kModeOpt:
  case Options::kModeCompress:
  case Options::kModeTest: {
    printHeader();

    int err = 0;

    std::string out_file = options.archive_file.getName();
    File fout;

    if (options.mode == Options::kModeOpt) {
      std::cout << "Optimizing" << std::endl;
      uint64_t best_size = std::numeric_limits<uint64_t>::max();
      size_t best_var = 0;
      std::ofstream opt_file("opt_result.txt");
      // static const size_t kOpts = 10624;
      static const size_t kOpts = 3;
      // size_t opts[kOpts] = {0,1,2,3,15,14,4,6,7,8,9,17,12,11,13,5,10,18,20,19,21,26,22,28,23,24,16,25,27,29,31,32,36,33,34,35,37,30,38,39,};
      // size_t opts[kOpts] = {}; for (size_t i = 0; i < kOpts; ++i) opts[i] = i;
      size_t opts[kOpts] = {};
      //size_t opts[] = {7,14,1,12,3,4,11,15,9,16,5,6,18,13,19,30,45,20,21,22,23,17,8,2,26,10,32,43,36,35,42,29,34,24,25,37,31,33,39,38,0,41,28,40,44,58,46,59,92,27,60,61,91,63,95,47,64,124,94,62,93,96,123,125,72,69,65,67,83,68,66,73,82,70,80,76,71,81,77,87,78,74,79,84,75,48,49,50,51,52,53,54,55,56,57,86,88,97,98,99,100,85,101,90,103,104,89,105,107,102,108,109,110,111,106,113,112,114,115,116,119,118,120,121,117,122,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,151,144,145,146,147,148,149,150,152,153,155,156,157,154,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,239,227,228,229,230,231,232,233,234,235,236,237,238,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,};
      if (false) {
        auto temp = ReadCSI<size_t>("optin.txt");
        check(temp.size() == kOpts);
        std::copy(temp.begin(), temp.end(), &opts[0]);
      }
      size_t best_opts[kOpts] = {};
      srand(clock() + 291231);
      size_t cur_index = 0;
      size_t len = 1;
      // size_t kMaxIndex = 128 - len;
      // size_t kMaxIndex = cm::kModelCount;
      size_t kMaxIndex = 12345;
      size_t bads = 0;
      size_t best_cur = 0;
      static constexpr size_t kAvgCount = 2;
      double total[kAvgCount] = {};
      size_t count[kAvgCount] = {};
      double min_time = std::numeric_limits<double>::max();
      const bool kPerm = false;
      if (kPerm) {
        for (size_t i = 0;; ++i) {
          if ((i & 255) == 255 && false) {
            len -= len > 1;
            kMaxIndex = 128 - len;
          }
          auto a = i % kMaxIndex;
          auto b = (i + 1 + std::rand() % (kMaxIndex - 1)) % kMaxIndex;
          VoidWriteStream fout;
          Archive archive(&fout, options.options_);
          if (i != 0) {
            ReplaceSubstring(opts, a, len, b, kOpts);
          }
          if (!archive.setOpts(opts)) {
            std::cerr << "Failed to set opts" << std::endl;
            continue;
          }
          uint64_t in_bytes = archive.compress(options.files);
          if (in_bytes == 0) continue;
          const auto size = fout.tell();
          std::cout << i << ": swap " << a << " to " << b << " " << size << std::endl;
          if (size < best_size) {
            std::cout << "IMPROVEMENT " << i << ": " << size << std::endl;
            opt_file << i << ": " << size << " ";
            for (auto opt : opts) opt_file << opt << ",";
            opt_file << std::endl;
            std::copy_n(opts, kOpts, best_opts);
            best_size = size;
          } else {
            std::copy_n(best_opts, kOpts, opts);
          }
        }
      } else {
        for (auto o : opts) check(o <= kMaxIndex);
        for (size_t i = 0;; ++i) {
          const clock_t start = clock();
          VoidWriteStream fout;
          Archive archive(&fout, options.options_);
          if (!archive.setOpts(opts)) {
            continue;
          }
          uint64_t in_bytes = archive.compress(options.files);
          if (in_bytes == 0) continue;
          const double time = clockToSeconds(clock() - start);
          total[i % kAvgCount] += time;
          min_time = std::min(min_time, time);
          const auto size = fout.tell();
          opt_file << "opts ";
          for (auto opt : opts) opt_file << opt << ",";
          ++count[i % kAvgCount];
          auto before_index = cur_index;
          auto before_opt = opts[before_index];
          if (size < best_size) {
            best_size = size;
            std::copy_n(opts, kOpts, best_opts);
            best_var = opts[cur_index];
            bads = 0;
          } 
          if (opts[cur_index] >= kMaxIndex) {
            std::copy_n(best_opts, kOpts, opts);
            cur_index = (cur_index + 1) % kOpts;
            opts[cur_index] = 0;
          } else {
            ++opts[cur_index];
          }

          std::ostringstream ss;
          double avgs[kAvgCount] = {};
          for (size_t i = 0; i < kAvgCount; ++i) {
            if (count[i] != 0) avgs[i] = total[i] / double(count[i]);
          }
          double avg = std::accumulate(total, total + kAvgCount, 0.0) / double(std::accumulate(count, count + kAvgCount, 0u));
          ss << " -> " << formatNumber(size) << " best " << best_var << " in " << time << "s avg "
             << avg << "(";
          for (double d : avgs) ss << d << ",";
          ss << ") min " << min_time;

          opt_file << ss.str() << std::endl << std::flush;

          std::cout << "opt[" << before_index << "]=" << before_opt << " best=" << best_var << "(" << formatNumber(best_size) << ") "
            << formatNumber(in_bytes) << ss.str() << std::endl;
        }
      }
    } else {
      const clock_t start = clock();
      if (err = fout.open(out_file, std::ios_base::out | std::ios_base::binary)) {
        std::cerr << "Error opening: " << out_file << " (" << errstr(err) << ")" << std::endl;
        return 2;
      }

      std::cout << "Compressing to " << out_file << " mode=" << options.options_.comp_level_ << " mem=" << options.options_.mem_usage_ << std::endl;
      Archive archive(&fout, options.options_);
      uint64_t in_bytes = archive.compress(options.files);
      clock_t time = clock() - start;
      std::cout << "Done compressing " << formatNumber(in_bytes) << " -> " << formatNumber(fout.tell())
        << " in " << std::setprecision(3) << clockToSeconds(time) << "s"
        << " bpc=" << double(fout.tell()) * 8.0 / double(in_bytes) << std::endl;

      fout.close();

      if (options.mode == Options::kModeSingleTest) {
        if (err = fout.open(out_file, std::ios_base::in | std::ios_base::binary)) {
          std::cerr << "Error opening: " << out_file << " (" << errstr(err) << ")" << std::endl;
          return 1;
        }
        Archive archive(&fout);
        archive.list();
        std::cout << "Verifying archive decompression" << std::endl;
        archive.decompress("", true);
      }
    }
    break;
  }
  case Options::kModeAdd: {
    // Add a single file.
    break;
  }
  case Options::kModeList: {
    auto in_file = options.archive_file.getName();
    File fin;
    int err = 0;
    if (err = fin.open(in_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening: " << in_file << " (" << errstr(err) << ")" << std::endl;
      return 1;
    }
    printHeader();
    std::cout << "Listing files in archive " << in_file << std::endl;
    Archive archive(&fin);
    const auto& header = archive.getHeader();
    if (!header.isArchive()) {
      std::cerr << "Attempting to open non mcm compatible file" << std::endl;
      return 1;
    }
    if (!header.isSameVersion()) {
      std::cerr << "Attempting to open old version " << header.majorVersion() << "." << header.minorVersion() << std::endl;
      return 1;
    }
    archive.list();
    fin.close();
    break;
  }
  case Options::kModeDecompress: {
    auto in_file = options.archive_file.getName();
    File fin;
    File fout;
    int err = 0;
    if (err = fin.open(in_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening: " << in_file << " (" << errstr(err) << ")" << std::endl;
      return 1;
    }
    printHeader();
    std::cout << "Decompresing archive " << in_file << std::endl;
    Archive archive(&fin);
    const auto& header = archive.getHeader();
    if (!header.isArchive()) {
      std::cerr << "Attempting to decompress non archive file" << std::endl;
      return 1;
    }
    if (!header.isSameVersion()) {
      std::cerr << "Attempting to decompress other version " << header.majorVersion() << "." << header.minorVersion() << std::endl;
      return 1;
    }
    // archive.decompress(options.files.back().getName());
    archive.decompress("");
    fin.close();
    // Decompress the single file in the archive to the output out.
    break;
  }
  case Options::kModeExtract: {
    // Extract a single file from multi file archive.
    break;
  }
  case Options::kModeExtractAll: {
    // Extract all the files in the archive.
    // archive.ExtractAll();
    break;
  }
  case Options::kModeObserver: {
    printHeader();
    std::cout << "Running in Observer Mode" << std::endl;
    // Read the input file
    std::vector<FileInfo> files = options.files;
    uint64_t total_size = 0;
    for (const auto& f : files) {
      File fin;
      if (fin.open(f.getName(), std::ios_base::in | std::ios_base::binary)) {
        std::cerr << "Error opening: " << f.getName() << std::endl;
        return 1;
      }
      std::cout << "Length of " << f.getName() << ": " << fin.length() << std::endl;
      total_size += fin.length();
      fin.close();
    }
    std::cout << "Total size " << total_size << std::endl;
    // Create a buffer for the file
    std::vector<uint8_t> buffer(total_size);
    uint64_t pos = 0;
    for (const auto& f : files) {
      File fin(f.getName(), std::ios_base::in | std::ios_base::binary);
      size_t count = fin.read(buffer.data() + pos, buffer.size() - pos);
      std::cout << "Read " << count << " bytes from " << f.getName() << std::endl;
      pos += count;
    }
    std::cout << "Read " << pos << " bytes" << std::endl;
    // Create CM
    cm::CM<8, true> compressor(FrequencyCounter<256>(), 8, true, Detector::kProfileSimple);
    compressor.cur_profile_ = cm::CMProfile::CreateSimple(8);
    compressor.observer_mode = true;
    // Create streams
    ReadMemoryStream rms(buffer.data(), buffer.data() + buffer.size());
    VoidWriteStream vws;
    std::cout << "Starting compress" << std::endl;
    std::cout << "Compressing " << buffer.size() << " bytes" << std::endl;
    compressor.compress(&rms, &vws, buffer.size());
    std::cout << "entropies.size() = " << compressor.entropies.size() << std::endl;
    std::cout << "Compress done" << std::endl;
    // Skip stock compression size for observer to avoid hang
    uint64_t stock_size = 0;
    // Output entropies
    std::string out_file = options.archive_file.getName();
    if (out_file.empty()) {
      out_file = files[0].getName() + ".entropy";
    }
    std::ofstream ofs(out_file, std::ios::binary);
    if (!ofs) {
      std::cerr << "Error opening output file: " << out_file << std::endl;
      return 1;
    }
    // Write total bytes
    uint64_t num_bytes = compressor.entropies.size();
    ofs.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    // Write stock size
    ofs.write(reinterpret_cast<const char*>(&stock_size), sizeof(stock_size));
    // Write entropies
    ofs.write(reinterpret_cast<const char*>(compressor.entropies.data()), num_bytes * sizeof(double));
    ofs.close();
    std::cout << "Wrote " << num_bytes << " entropy values and stock size " << stock_size << " to " << out_file << std::endl;
    break;
  }
  case Options::kModeSegment: {
    printHeader();
    std::cout << "Running Segmentation Mode" << std::endl;
    // Read .entropy file
    std::vector<FileInfo> files = options.files;
    std::string in_file = files[0].getName();
    std::ifstream ifs(in_file, std::ios::binary);
    if (!ifs) {
      std::cerr << "Error opening entropy file: " << in_file << std::endl;
      return 1;
    }
    uint64_t num_bytes;
    ifs.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));
    uint64_t stock_size;
    ifs.read(reinterpret_cast<char*>(&stock_size), sizeof(stock_size));
    std::vector<double> entropies(num_bytes);
    ifs.read(reinterpret_cast<char*>(entropies.data()), num_bytes * sizeof(double));
    ifs.close();
    std::cout << "Read " << formatNumber(num_bytes) << " entropy values, stock size " << formatNumber(stock_size) << std::endl;

    // Compute entropy stats
    double min_e = *std::min_element(entropies.begin(), entropies.end());
    double max_e = *std::max_element(entropies.begin(), entropies.end());
    double avg_e = std::accumulate(entropies.begin(), entropies.end(), 0.0) / entropies.size();
    std::cout << "Entropy stats: min=" << min_e << " max=" << max_e << " avg=" << avg_e << std::endl;

    // Smoothing: moving average
    const size_t window = options.segment_window;
    std::vector<double> smoothed(num_bytes);
    for (size_t i = 0; i < num_bytes; ++i) {
      float sum = 0;
      size_t count = 0;
      for (size_t j = (i > window/2 ? i - window/2 : 0); j < std::min(i + window/2 + 1, num_bytes); ++j) {
        sum += entropies[j];
        ++count;
      }
      smoothed[i] = sum / count;
    }

    // Compute smoothed stats
    double min_s = *std::min_element(smoothed.begin(), smoothed.end());
    double max_s = *std::max_element(smoothed.begin(), smoothed.end());
    double avg_s = std::accumulate(smoothed.begin(), smoothed.end(), 0.0) / smoothed.size();
    std::cout << "Smoothed stats: min=" << min_s << " max=" << max_s << " avg=" << avg_s << std::endl;

    // Boundary detection: find points where smoothed entropy > threshold
    const double threshold = options.segment_threshold;  // Minimum entropy value
    const size_t min_segment = options.segment_min_segment;
    std::vector<size_t> boundaries;
    boundaries.push_back(0);
    for (size_t i = 1; i < num_bytes; ++i) {
      if (smoothed[i] > threshold && i - boundaries.back() >= min_segment) {
        boundaries.push_back(i);
      }
    }
    boundaries.push_back(num_bytes);

    // Atomic Fusion: Hot/Cold Dependency Check
    std::string original_file;
    if (in_file.find("enwik8_5MB") != std::string::npos) {
      original_file = "testFiles/enwik8_5MB";
    } else if (in_file.find("helloWorld") != std::string::npos) {
      original_file = "testFiles/helloWorld.txt";
    } else {
      original_file = in_file.substr(0, in_file.size() - 7);
    }
    File original_fin(original_file, std::ios_base::in | std::ios_base::binary);
    if (!original_fin.isOpen()) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    uint64_t file_size = original_fin.length();
    if (num_bytes > file_size) {
      num_bytes = file_size;
      entropies.resize(num_bytes);
    }
    float global_avg = 0;
    for (float e : entropies) global_avg += e;
    global_avg /= num_bytes;
    std::vector<size_t> to_remove;
/*    for (size_t i = 1; i < boundaries.size() - 1; ++i) {
      size_t start_b = boundaries[i];
      size_t end_b = boundaries[i + 1];
      size_t len_b = end_b - start_b;
      float avg_b = 0;
      for (size_t j = start_b; j < end_b; ++j) avg_b += entropies[j];
      avg_b /= len_b;
      if (avg_b >= global_avg) continue; // skip high-entropy segments
      // Read segment bytes
      std::vector<uint8_t> segment_bytes(len_b);
      original_fin.seek(start_b);
      size_t read_count = original_fin.read(segment_bytes.data(), len_b);
      if (read_count != len_b) {
        std::cerr << "Error reading segment bytes" << std::endl;
        continue;
      }
      // Cold compression
      cm::CM<12, true, uint32_t> cold_comp(FrequencyCounter<256>(), options.options_.mem_usage_, true, Detector::kProfileText);
      ReadMemoryStream rms_seg(segment_bytes.data(), segment_bytes.data() + len_b);
      VoidWriteStream vws_seg;
      cold_comp.compress(&rms_seg, &vws_seg, len_b);
      uint64_t cold_bits = vws_seg.tell() * 8;
      // Hot cost
      float hot_sum = 0;
      for (size_t j = start_b; j < end_b; ++j) hot_sum += entropies[j];
      // Check dependency
      if (cold_bits > hot_sum * 1.05f) {
        to_remove.push_back(i);
      }
    }
*/
    // Remove boundaries (fuse)
    for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
      boundaries.erase(boundaries.begin() + *it);
    }

    // Output segments
    std::string out_file = options.archive_file.getName();
    if (out_file.empty()) {
      out_file = in_file + ".segments";
    }
    std::ofstream ofs(out_file);
    if (!ofs) {
      std::cerr << "Error opening output file: " << out_file << std::endl;
      return 1;
    }
    uint64_t num_segments = boundaries.size() - 1;
    ofs << num_segments << std::endl;
    for (size_t i = 0; i < num_segments; ++i) {
      uint64_t start = boundaries[i];
      uint64_t length = boundaries[i+1] - boundaries[i];
      ofs << start << " " << length << std::endl;
    }
    ofs.close();
    std::cout << "Wrote " << formatNumber(num_segments) << " segments to " << out_file << std::endl;
    break;
  }
  case Options::kModeFingerprint: {
    printHeader();
    std::cout << "Running Fingerprinting Mode" << std::endl;
    // Read .segments file
    if (argc != 3) {
      std::cerr << "Usage: mcm -fingerprint <segments_file>" << std::endl;
      return 1;
    }
    std::string in_file = argv[2];
    std::cout << "in_file: " << in_file << std::endl;
    std::ifstream ifs(in_file);
    if (!ifs) {
      std::cerr << "Error opening segments file: " << in_file << std::endl;
      return 1;
    }
    size_t num_segments;
    ifs >> num_segments;
    std::vector<std::pair<size_t, size_t>> segments(num_segments);
    for (size_t i = 0; i < num_segments; ++i) {
      ifs >> segments[i].first >> segments[i].second;
    }
    ifs.close();

    // Determine original file
    std::string original_file;
    size_t entropy_pos = in_file.find(".entropy");
    if (entropy_pos != std::string::npos) {
      original_file = in_file.substr(0, entropy_pos);
    } else {
      original_file = in_file.substr(0, in_file.find_last_of('.'));
    }

    // Load original file
    File fin;
    if (fin.open(original_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    std::vector<uint8_t> file_data(fin.length());
    fin.read(&file_data[0], fin.length());
    fin.close();

    // Compute matrices
    std::vector<std::pair<size_t, size_t>> valid_segments;
    for (auto& seg : segments) {
      size_t start = seg.first;
      size_t len = seg.second;
      if (start >= file_data.size() || len == 0) continue;
      if (start + len > file_data.size()) {
        len = file_data.size() - start;
      }
      valid_segments.push_back({start, len});
    }
    size_t num_valid = valid_segments.size();
    std::vector<MarkovMatrix> matrices(num_valid);
    for (size_t i = 0; i < num_valid; ++i) {
      size_t start = valid_segments[i].first;
      size_t len = valid_segments[i].second;
      std::vector<uint8_t> segment_data(file_data.begin() + start, file_data.begin() + start + len);
      matrices[i] = compute_matrix(segment_data);
    }

    // Compute distances and find top-k
    size_t top_k = options.fingerprint_top_k;
    std::vector<std::vector<size_t>> candidates(num_valid);
    for (size_t i = 0; i < num_valid; ++i) {
      std::vector<std::pair<float, size_t>> distances;
      for (size_t j = 0; j < num_valid; ++j) {
        if (i == j) continue;
        float dist = hellinger_distance(matrices[i], matrices[j]);
        distances.emplace_back(dist, j);
      }
      std::sort(distances.begin(), distances.end());
      for (size_t k = 0; k < std::min(top_k, distances.size()); ++k) {
        candidates[i].push_back(distances[k].second);
      }
    }

    // Output .candidates file
    std::string out_file = in_file + ".candidates";
    std::ofstream ofs(out_file);
    if (!ofs) {
      std::cerr << "Error opening output file: " << out_file << std::endl;
      return 1;
    }
    for (size_t i = 0; i < num_valid; ++i) {
      ofs << i << ":";
      for (size_t j = 0; j < candidates[i].size(); ++j) {
        if (j > 0) ofs << ",";
        ofs << candidates[i][j];
      }
      ofs << std::endl;
    }
    ofs.close();
    std::cout << "Wrote candidate lists for " << num_valid << " segments to " << out_file << std::endl;
    break;
  }
  {
  case Options::kModeOracle: {
    printHeader();
    std::cout << "Running Oracle Mode" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
      std::cerr << "Usage: mcm -oracle <candidates_file>" << std::endl;
      return 1;
    }
    std::string in_file = argv[2];
    std::cout << "in_file: " << in_file << std::endl;
    std::ifstream ifs(in_file);
    if (!ifs) {
      std::cerr << "Error opening candidates file: " << in_file << std::endl;
      return 1;
    }
    std::vector<std::vector<size_t>> candidates;
    std::string line;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::string token;
      std::getline(iss, token, ':');
      size_t i = std::stoul(token);
      if (i >= candidates.size()) candidates.resize(i + 1);
      std::vector<size_t>& cands = candidates[i];
      while (std::getline(iss, token, ',')) {
        if (!token.empty()) cands.push_back(std::stoul(token));
      }
    }
    ifs.close();
    size_t num_segments = candidates.size();
    std::cout << "Read " << num_segments << " segments from candidates" << std::endl;

    // Determine segments file
    std::string segments_file = in_file.substr(0, in_file.find_last_of('.')); // remove .candidates
    std::cout << "segments_file: " << segments_file << std::endl;
    std::ifstream seg_ifs(segments_file);
    if (!seg_ifs) {
      std::cerr << "Error opening segments file: " << segments_file << std::endl;
      return 1;
    }
    size_t num_seg;
    seg_ifs >> num_seg;
    std::vector<std::pair<size_t, size_t>> segments(num_seg);
    for (size_t i = 0; i < num_seg; ++i) {
      seg_ifs >> segments[i].first >> segments[i].second;
    }
    seg_ifs.close();

    // Load original file
    std::string original_file = segments_file.substr(0, segments_file.find_last_of('.'));
    std::cout << "original_file: " << original_file << std::endl;
    File fin;
    if (fin.open(original_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    size_t file_size = fin.length();
    std::vector<uint8_t> file_data(file_size);
    fin.read(&file_data[0], file_size);
    fin.close();

    // Filter valid segments as in fingerprint
    std::vector<std::pair<size_t, size_t>> valid_segments;
    for (auto& seg : segments) {
      size_t start = seg.first;
      size_t len = seg.second;
      if (start >= file_size || len == 0) continue;
      if (start + len > file_size) {
        len = file_size - start;
      }
      valid_segments.push_back({start, len});
    }
    if (valid_segments.size() != num_segments) {
      std::cerr << "Mismatch: valid_segments " << valid_segments.size() << " vs candidates " << num_segments << std::endl;
      return 1;
    }

    // Build reverse map: pred -> list of succ that have pred as candidate
    std::map<size_t, std::vector<size_t>> pred_to_succ;
    for (size_t succ = 0; succ < num_segments; ++succ) {
      for (size_t pred : candidates[succ]) {
        pred_to_succ[pred].push_back(succ);
      }
    }
    std::cout << "pred_to_succ size: " << pred_to_succ.size() << std::endl << std::flush;

    // For each pred, compress pred once, take snapshot, then evaluate all succ that have pred as candidate
    std::vector<std::vector<std::pair<size_t, double>>> pred_costs(num_segments);  // for each pred, list of (succ, cost)
    std::cout << "Number of pred to process: " << pred_to_succ.size() << std::endl << std::flush;

    // Use multi-process approach
    int ret = run_oracle_multiprocess(argv[0], file_data, valid_segments, pred_to_succ, pred_costs);
    if (ret != 0) {
      std::cerr << "Multi-process oracle failed" << std::endl;
      return ret;
    }

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
    std::cout << "Oracle processing completed in " << elapsed.count() << " seconds" << std::endl;
    break;
  }
  case Options::kModePathCover: {
    printHeader();
    std::cout << "Running PathCover Mode" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
      std::cerr << "Usage: mcm -pathcover <oracle_file>" << std::endl;
      return 1;
    }
    std::string in_file = argv[2];
    std::cout << "in_file: " << in_file << std::endl;
    // Read segments file
    std::string segments_file = in_file.substr(0, in_file.find(".candidates"));
    std::cout << "segments_file: " << segments_file << std::endl;
    std::ifstream seg_ifs(segments_file);
    if (!seg_ifs) {
      std::cerr << "Error opening segments file: " << segments_file << std::endl;
      return 1;
    }
    size_t num_seg;
    seg_ifs >> num_seg;
    std::vector<std::pair<size_t, size_t>> segments(num_seg);
    for (size_t i = 0; i < num_seg; ++i) {
      seg_ifs >> segments[i].first >> segments[i].second;
    }
    seg_ifs.close();
    size_t num_segments = num_seg;
    std::cout << "Read " << num_segments << " segments from segments file" << std::endl;

    std::string original_file = segments_file.substr(0, segments_file.find_last_of('.')); // remove .segments
    std::cout << "original_file: " << original_file << std::endl;

    // Read entropy for segment costs
    std::string entropy_file = original_file + ".entropy";
    std::vector<double> entropies;
    std::ifstream eifs(entropy_file, std::ios::binary);
    if (eifs) {
      uint64_t num_bytes;
      eifs.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));
      entropies.resize(num_bytes);
      eifs.read(reinterpret_cast<char*>(&entropies[0]), num_bytes * sizeof(double));
      eifs.close();
      std::cout << "Read " << num_bytes << " entropies" << std::endl;
    } else {
      std::cout << "Warning: entropy file not found, using length for costs" << std::endl;
    }

    // Read candidates for orphan clustering
    std::string candidates_file = segments_file + ".candidates";
    std::cout << "candidates_file: " << candidates_file << std::endl;
    std::ifstream cifs(candidates_file);
    std::vector<std::vector<std::pair<size_t, double>>> candidate_distances(num_segments);
    if (cifs) {
      std::string line;
      while (std::getline(cifs, line)) {
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ':');
        size_t i = std::stoul(token);
        if (i >= num_segments) continue;
        std::vector<std::pair<size_t, double>>& dists = candidate_distances[i];
        while (std::getline(iss, token, ',')) {
          if (token.empty()) continue;
          size_t j = std::stoul(token);
          std::getline(iss, token, ',');
          double d = std::stod(token);
          dists.push_back({j, d});
        }
      }
      cifs.close();
      std::cout << "Read candidate distances" << std::endl;
    } else {
      std::cout << "Warning: candidates file not found, using simple orphan sort" << std::endl;
    }

    // Read oracle
    std::vector<size_t> best_j(num_segments, std::numeric_limits<size_t>::max());
    std::vector<double> costs(num_segments, 1e100);
    std::ifstream oracle_ifs(in_file);
    if (!oracle_ifs) {
      std::cerr << "Error opening oracle file: " << in_file << std::endl;
      return 1;
    }
    std::string line;
    while (std::getline(oracle_ifs, line)) {
      std::istringstream iss(line);
      std::string token;
      std::getline(iss, token, ':');
      size_t i = std::stoul(token);
      if (i >= num_segments) continue;
      std::getline(iss, token, ',');
      best_j[i] = std::stoul(token);
      std::getline(iss, token);
      costs[i] = std::stod(token);
    }
    oracle_ifs.close();
    std::cout << "Read oracle data" << std::endl;

    // Load original file
    File fin;
    if (fin.open(original_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    size_t file_size = fin.length();
    std::vector<uint8_t> file_data(file_size);
    fin.read(&file_data[0], file_size);
    fin.close();

    // Collect edges
    struct Edge {
      size_t from, to;
      double benefit;
    };
    std::vector<Edge> edges;
    for (size_t i = 0; i < num_segments; ++i) {
      if (best_j[i] != std::numeric_limits<size_t>::max()) {
        size_t len_j = segments[best_j[i]].second;
        double benefit = -costs[i] / len_j;  // Normalize by target segment length to prefer longer segments
        edges.push_back({i, best_j[i], benefit});
      }
    }
    // Sort edges by benefit descending
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
      return a.benefit > b.benefit;
    });

    // Initialize chains
    std::vector<std::vector<size_t>> chains(num_segments);
    std::vector<size_t> chain_of(num_segments);
    for (size_t i = 0; i < num_segments; ++i) {
      chains[i] = {i};
      chain_of[i] = i;
    }

    // Greedy linking
    for (const auto& edge : edges) {
      size_t from_chain = chain_of[edge.from];
      size_t to_chain = chain_of[edge.to];
      if (from_chain != to_chain && chains[from_chain].back() == edge.from && chains[to_chain].front() == edge.to) {
        // Merge: append to_chain to from_chain
        chains[from_chain].insert(chains[from_chain].end(), chains[to_chain].begin(), chains[to_chain].end());
        for (size_t seg : chains[to_chain]) {
          chain_of[seg] = from_chain;
        }
        chains[to_chain].clear();
      }
    }

    // Collect valid chains and orphans
    std::vector<std::vector<size_t>> valid_chains;
    std::vector<size_t> orphans;
    for (size_t i = 0; i < num_segments; ++i) {
      if (!chains[i].empty()) {
        if (chains[i].size() > 1) {
          valid_chains.push_back(chains[i]);
        } else {
          orphans.push_back(chains[i][0]);
        }
      }
    }

    // Compute densities for chains
    struct ChainInfo {
      std::vector<size_t> segments;
      double density;
    };
    std::vector<ChainInfo> chain_infos;
    for (const auto& chain : valid_chains) {
      double total_benefit = 0.0;
      size_t total_bytes = 0;
      for (size_t seg : chain) {
        total_bytes += segments[seg].second;
        // Benefit from edges: for each transition in chain
        if (seg < chain.size() - 1) {
          size_t next = chain[seg + 1];
          // Find the edge benefit
          for (const auto& e : edges) {
            if (e.from == seg && e.to == next) {
              total_benefit += e.benefit;
              break;
            }
          }
        }
      }
      double density = total_benefit / total_bytes;
      chain_infos.push_back({chain, density});
    }
    // Sort chains by density descending
    std::sort(chain_infos.begin(), chain_infos.end(), [](const ChainInfo& a, const ChainInfo& b) {
      return a.density > b.density;
    });

    // Cluster orphans
    std::vector<std::vector<size_t>> orphan_clusters;
    if (!candidate_distances.empty() && !candidate_distances[0].empty()) {
      // Build graph for orphans
      std::vector<std::vector<size_t>> graph(orphans.size());
      std::map<size_t, size_t> orphan_index;
      for (size_t idx = 0; idx < orphans.size(); ++idx) {
        orphan_index[orphans[idx]] = idx;
      }
      const double threshold = 0.5; // Hellinger distance threshold
      for (size_t i = 0; i < orphans.size(); ++i) {
        size_t seg_i = orphans[i];
        for (const auto& p : candidate_distances[seg_i]) {
          size_t j = p.first;
          double d = p.second;
          if (d < threshold && orphan_index.count(j)) {
            size_t idx_j = orphan_index[j];
            if (std::find(graph[i].begin(), graph[i].end(), idx_j) == graph[i].end()) {
              graph[i].push_back(idx_j);
              graph[idx_j].push_back(i); // undirected
            }
          }
        }
      }
      // Find connected components using BFS
      std::vector<bool> visited(orphans.size(), false);
      for (size_t i = 0; i < orphans.size(); ++i) {
        if (!visited[i]) {
          std::vector<size_t> cluster;
          std::queue<size_t> q;
          q.push(i);
          visited[i] = true;
          while (!q.empty()) {
            size_t idx = q.front(); q.pop();
            cluster.push_back(orphans[idx]);
            for (size_t nei : graph[idx]) {
              if (!visited[nei]) {
                visited[nei] = true;
                q.push(nei);
              }
            }
          }
          orphan_clusters.push_back(cluster);
        }
      }
    } else {
      orphan_clusters = {orphans};
    }

    // Sort within each cluster by entropy cost ascending
    auto get_cost = [&](size_t seg) -> double {
      if (!entropies.empty()) {
        double cost = 0.0;
        size_t start = segments[seg].first;
        size_t len = segments[seg].second;
        for (size_t k = 0; k < len; ++k) {
          cost += entropies[start + k];
        }
        return cost;
      } else {
        return segments[seg].second; // length as proxy
      }
    };
    for (auto& cluster : orphan_clusters) {
      std::sort(cluster.begin(), cluster.end(), [&](size_t a, size_t b) {
        return get_cost(a) < get_cost(b);
      });
    }

    // Flatten orphan clusters
    std::vector<size_t> sorted_orphans;
    for (const auto& cluster : orphan_clusters) {
      sorted_orphans.insert(sorted_orphans.end(), cluster.begin(), cluster.end());
    }

    // Build reordered segment list
    std::vector<size_t> reordered_segments;
    for (const auto& ci : chain_infos) {
      reordered_segments.insert(reordered_segments.end(), ci.segments.begin(), ci.segments.end());
    }
    reordered_segments.insert(reordered_segments.end(), sorted_orphans.begin(), sorted_orphans.end());

    // Generate reordered file
    std::string reordered_file = original_file + ".reordered";
    std::ofstream rofs(reordered_file, std::ios::binary);
    if (!rofs) {
      std::cerr << "Error opening reordered file: " << reordered_file << std::endl;
      return 1;
    }
    for (size_t seg : reordered_segments) {
      size_t start = segments[seg].first;
      size_t len = segments[seg].second;
      rofs.write(reinterpret_cast<const char*>(&file_data[start]), len);
    }
    rofs.close();
    std::cout << "Wrote reordered file to " << reordered_file << std::endl;

    // Generate segment table (lengths in reordered order)
    std::string segment_table_file = reordered_file + ".segments";
    std::ofstream stofs(segment_table_file);
    if (!stofs) {
      std::cerr << "Error opening segment table file: " << segment_table_file << std::endl;
      return 1;
    }
    for (size_t seg : reordered_segments) {
      stofs << segments[seg].second << std::endl;
    }
    stofs.close();
    std::cout << "Wrote segment table to " << segment_table_file << std::endl;

    // Generate RLE-move index (segment ids in reordered order, RLE-encoded)
    std::string index_file = reordered_file + ".index";
    std::ofstream iofs(index_file);
    if (!iofs) {
      std::cerr << "Error opening index file: " << index_file << std::endl;
      return 1;
    }
    if (!reordered_segments.empty()) {
      size_t current = reordered_segments[0];
      size_t count = 1;
      for (size_t i = 1; i < reordered_segments.size(); ++i) {
        if (reordered_segments[i] == current) {
          count++;
        } else {
          iofs << current << "," << count << std::endl;
          current = reordered_segments[i];
          count = 1;
        }
      }
      iofs << current << "," << count << std::endl;
    }
    iofs.close();
    std::cout << "Wrote RLE index to " << index_file << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Phase 5 completed in " << elapsed.count() << " seconds" << std::endl;

    break;
  }
  }
  }
  return 0;
}