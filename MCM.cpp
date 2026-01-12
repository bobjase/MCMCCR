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
#include <eh.h>
#include <deque>
#include <vector>
#include <array>
#include <algorithm>
#include <io.h>
#include <fcntl.h>
#include <cctype>

// --- Helper: Binary I/O ---
#include "Util.hpp"
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
#include <atomic>
#include <mutex>
#include <cmath>
#include <numeric>

#include <omp.h>

#include <fstream>

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

#ifdef NDEBUG
static constexpr bool kReleaseBuild = true;
#else
static constexpr bool kReleaseBuild = false;
#endif

static constexpr bool calibration_mode = false;

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

// --- Hybrid Holographic Fingerprinting Structures ---

struct Fingerprint {
    // Tier 1: Vocabulary (Asymmetric Containment)
    std::vector<uint32_t> minhashes; // Bottom-256 hashes

    // Tier 2A: Structural Energy (WHT Spectrum)
    std::vector<float> wht_energy;   // Energy in 16 dyadic bands

    // Tier 2B: Structural Alignment (GCD Signature)
    std::vector<int> gcd_peaks;      // Top 3 common divisors of symbol gaps

    // Tier 3: State Volatility
    float volatility;                // Entropy estimate
};

struct DistanceComponents {
    float vocab;
    std::vector<float> wht;
    float gcd;
    float vol;
};

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

static void printHeader() {
  std::cout << "MCM compressor" << std::endl;
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
  size_t segment_window = 256;
  float segment_threshold = 0.0f;
  size_t segment_min_segment = 4096;
  size_t segment_max_segment = 65536;  // 64KB max segment size
  size_t segment_lookback = 1024;  // Fingerprinting parameters
  size_t fingerprint_top_k = 4096;
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
      << "-max-segment <size> maximum segment size (default 65536)" << std::endl
      << "-lookback <size> lookback distance for inflection points (default 512)" << std::endl
      // << "-b <mb> specifies block size in MB" << std::endl
      // << "-t <threads> the number of threads to use (decompression requires the same number of threads" << std::endl
      << "Examples:" << std::endl
      << "Compress: " << name << " -m9 enwik8 enwik8.mcm" << std::endl
      << "Decompress: " << name << " d enwik8.mcm enwik8.ref" << std::endl;
    std::cout << std::flush;
    return 0;
  }

  int parse(int argc, char* argv[]) {
    assert(argc >= 1);
    std::string program(trimExt(argv[0]));
    if (argc <= 1) {
      return usage(program);
    }
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
      } else if (arg == "-max-segment") {
        if (i + 1 >= argc) return usage(program);
        segment_max_segment = std::stoull(argv[++i]);
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

// SEH translator
void se_translator(unsigned int code, EXCEPTION_POINTERS* ep) {
    throw std::runtime_error("SEH exception");
}

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

        // Memory map segments file
        HANDLE hSegFile = CreateFileA(segments_file.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hSegFile == INVALID_HANDLE_VALUE) {
            debugError("CreateFile failed for segments file: " + segments_file);
            return 1;
        }
        LARGE_INTEGER segFileSize;
        if (!GetFileSizeEx(hSegFile, &segFileSize)) {
            debugError("GetFileSizeEx failed for segments file");
            CloseHandle(hSegFile);
            return 1;
        }
        HANDLE hSegMapping = CreateFileMapping(hSegFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hSegMapping == NULL) {
            debugError("CreateFileMapping failed for segments");
            CloseHandle(hSegFile);
            return 1;
        }
        LPVOID pSegView = MapViewOfFile(hSegMapping, FILE_MAP_READ, 0, 0, 0);
        if (pSegView == NULL) {
            debugError("MapViewOfFile failed for segments");
            CloseHandle(hSegMapping);
            CloseHandle(hSegFile);
            return 1;
        }
        std::string seg_content((char*)pSegView, segFileSize.QuadPart);
        std::istringstream seg_iss(seg_content);
        uint64_t num_segments;
        seg_iss >> num_segments;
        debugLog("num_segments: " + std::to_string(num_segments));
        std::vector<std::pair<size_t, size_t>> valid_segments(num_segments);
        debugLog("valid_segments resized");
        for (auto& seg : valid_segments) {
            seg_iss >> seg.first >> seg.second;
        }
        debugLog("read valid_segments");
        // Unmap segments
        UnmapViewOfFile(pSegView);
        CloseHandle(hSegMapping);
        CloseHandle(hSegFile);
        debugLog("segments file unmapped");

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
            size_t start = p.first;
            size_t len = p.second;
            if (start >= file_size || len == 0) continue;
            if (start + len > file_size) {
                len = file_size - start;
                p.second = len;
            }
        }

        // Define CM objects ONCE (performance optimization)
        cm::CM<8, false> cm(FrequencyCounter<256>(), 0, false, Detector::kProfileSimple);
        // Use full profile for max compression accuracy
        cm.cur_profile_ = cm::CMProfile();
        for (int i = 0; i < static_cast<int>(cm::kModelCount); ++i) {
            cm.cur_profile_.EnableModel(static_cast<cm::ModelType>(i));
        }
        cm.cur_profile_.SetMatchModelOrder(12);
        cm.cur_profile_.SetMinLZPLen(10);
        cm.observer_mode = true;

        cm::CM<8, false> cm_alone(FrequencyCounter<256>(), 0, false, Detector::kProfileSimple);
        // Use full profile for max compression accuracy
        cm_alone.cur_profile_ = cm::CMProfile();
        for (int i = 0; i < static_cast<int>(cm::kModelCount); ++i) {
            cm_alone.cur_profile_.EnableModel(static_cast<cm::ModelType>(i));
        }
        cm_alone.cur_profile_.SetMatchModelOrder(12);
        cm_alone.cur_profile_.SetMinLZPLen(10);
        cm_alone.observer_mode = true;

        // Worker loop
        while (true) {
            // Read pred_id
            size_t pred_id;
            if (fread(&pred_id, sizeof(size_t), 1, stdin) != 1) {
                debugError("Failed to read pred_id, exiting");
                break;
            }
            debugLog("read pred_id: " + std::to_string(pred_id));
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
            size_t start_pred = valid_segments[pred_id].first;
            size_t len_pred = valid_segments[pred_id].second;
            std::vector<uint8_t> data_pred(len_pred);
            memcpy(data_pred.data(), file_data + start_pred, len_pred);
            debugLog("Got pred data");

            // Compute fingerprint for pred
            Fingerprint fp_pred = compute_fingerprint(data_pred);

            // --- FIX 1: RUN PREDECESSOR ---
            // We must populate the compressor state with the Predecessor's context
            
            // 1. Clear the Hash Table (Since we disabled the automatic wipe in restoreSnapshot)
            memset(cm.base_hash_table_, 0, cm.hash_alloc_size_);
            
            // 2. Init & Compress
            cm.init(); 
            cm.skip_init = true; 
            cm.SetOverlay(nullptr); 
            
            ReadMemoryStream in_pred(data_pred.data(), data_pred.data() + data_pred.size());
            VoidWriteStream out_pred; // <--- ADD THIS LINE HERE
            
            cm.compress(&in_pred, &out_pred, data_pred.size());

            // Take pred snapshot for tournament
            auto pred_snapshot = cm.takeSnapshot();
            double pred_cost = cm.getAccumulatedEntropy();

            // this is always outputting 0 -- seems like a bug
            //debugError("Pred cost: " + std::to_string(pred_cost));

            // 2. TOURNAMENT ROSTER (Fixed Capacity)
            struct Candidate {
                size_t id;
                cm::PagedOverlay* overlay;
                double savings_rate;
            };
            std::vector<Candidate> roster;
            roster.reserve(129); // Keep it small!

            // The "Working" Overlay (Recycled)
            cm::PagedOverlay* worker_ov = new cm::PagedOverlay(cm.base_hash_table_, cm.hash_alloc_size_);
            
            // Reusable Chunk Buffer
            std::vector<uint8_t> chunk_buf;
            chunk_buf.reserve(4096); 

            // Void output stream for compression
            //VoidWriteStream out_pred;

            size_t roundOneSize = 256;
            // --- ROUND 1: QUALIFIERS (Streaming & Recycling) ---
            for (size_t succ : succ_list) {
                if (succ >= valid_segments.size()) {
                  debugError("Invalid succ segment size");
                  continue;
                }

                // 1. Setup Chunk
                size_t start = valid_segments[succ].first;
                size_t len = valid_segments[succ].second;
                size_t scan_len = std::min((size_t)roundOneSize, len);

                // 2. Reset Worker (Reuse memory)
                worker_ov->Reset(); 

                // 3. Run Compression (ZERO COPY)
                cm.SetOverlay(worker_ov);
                cm.restoreSnapshot(pred_snapshot);
                cm.byte_index = 0;

                // Point directly to the memory-mapped file
                const uint8_t* ptr_start = (const uint8_t*)(file_data + start);
                ReadMemoryStream in_chunk(ptr_start, ptr_start + scan_len); // Uses Memory.hpp class
                
                //debugError("Starting compression for succ " + std::to_string(succ));

                cm.compress(&in_chunk, &out_pred, scan_len);

                //debugError("Finished compression for succ " + std::to_string(succ));

                // 4. Score
                double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                double alone_cost = succ_alone_costs[succ];
                
                double savings_rate = (alone_cost - joint_delta) / scan_len;

                // 5. King of the Hill Logic
                if (roster.size() < 128) {
                    // Always accept if we have room
                    roster.push_back({succ, worker_ov, savings_rate});
                    // Allocate a NEW worker for next turn (since we kept the old one)
                    worker_ov = new cm::PagedOverlay(cm.base_hash_table_, cm.hash_alloc_size_);
                } else {
                    // Find worst in roster
                    auto min_it = std::min_element(roster.begin(), roster.end(), 
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

            // Sort the final 32
            //std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ return a.savings_rate > b.savings_rate; });
            // Deterministic Sort: Break ties with ID
            std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ 
                if (std::abs(a.savings_rate - b.savings_rate) > 1e-9) return a.savings_rate > b.savings_rate;
                return a.id < b.id; // Tie-breaker
            });

            // --- ROUND 2: SEMI-FINALS (2048 Bytes) ---
            for (auto& cand : roster) {
                size_t len = valid_segments[cand.id].second;
                if (len <= roundOneSize) continue; // Already fully scanned
                
                size_t limit = 2048;
                size_t scan_len = std::min(len, limit);
                size_t start = valid_segments[cand.id].first;
                
                std::vector<uint8_t> chunk(scan_len);
                memcpy(chunk.data(), file_data + start, scan_len);

                // Reuse Overlay (it contains the hash writes from Round 1)
                cm.SetOverlay(cand.overlay);
                cm.restoreSnapshot(pred_snapshot); // Reset Mixers to start
                cm.byte_index = 0;
                
                // RESTART Scan (0 -> 2048)
                // Why restart? Because the Overlay preserved the Hash Table, but we reset the Mixers.
                // Re-running the first 256 bytes is negligible cost.
                MemoryReadStream in_chunk(chunk);
                cm.compress(&in_chunk, &out_pred, scan_len);

                double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                double alone_cost = succ_alone_costs[cand.id];
                
                cand.savings_rate = (alone_cost - joint_delta) / scan_len;
            }

            // Prune: Sort & Keep Top 16
            //std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ return a.savings_rate > b.savings_rate; });
            // Deterministic Sort
            std::sort(roster.begin(), roster.end(), [](const auto& a, const auto& b){ 
                if (std::abs(a.savings_rate - b.savings_rate) > 1e-9) return a.savings_rate > b.savings_rate;
                return a.id < b.id; 
            });
            
            size_t cut_2 = std::min(roster.size(), (size_t)64);
            for (size_t k = cut_2; k < roster.size(); ++k) delete roster[k].overlay;
            roster.resize(cut_2);

            // --- ROUND 3: FINALS (Full Run / 10KB Cap) ---
            for (const auto& cand : roster) {
                size_t len = valid_segments[cand.id].second;
                size_t scan_len = std::min(len, (size_t)10240); // Cap at 10KB
                
                std::vector<uint8_t> chunk(scan_len);
                memcpy(chunk.data(), file_data + valid_segments[cand.id].first, scan_len);

                cm.SetOverlay(cand.overlay);
                cm.restoreSnapshot(pred_snapshot);
                cm.byte_index = 0;
                
                MemoryReadStream in_chunk(chunk);
                cm.compress(&in_chunk, &out_pred, scan_len);

                double joint_delta = cm.getAccumulatedEntropy() - pred_cost;
                double alone_cost = succ_alone_costs[cand.id];
                
                // Result: Negative Cost = Savings
                double final_cost = joint_delta - alone_cost; 
                succ_costs[cand.id].emplace_back(pred_id, final_cost);
                
                delete cand.overlay; // Done with this candidate
            }
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
        return 1;
    } catch (const std::out_of_range& e) {
        return 1;
    } catch (const std::runtime_error& e) {
        return 1;
    } catch (const std::exception& e) {
        return 1;
    } catch (...) {
        return 1;
    }
}

int run_oracle_multiprocess(const char* exe_path, const char* in_file, const std::vector<uint8_t>& file_data, const std::vector<std::pair<size_t, size_t>>& valid_segments, const std::map<size_t, std::vector<size_t>>& pred_to_succ, std::vector<std::vector<std::pair<size_t, double>>>& pred_costs) {
    debugLog("run_oracle_multiprocess start");
    // Get full path
    char full_path[MAX_PATH];
    if (!GetFullPathNameA(in_file, MAX_PATH, full_path, NULL)) {
        std::cerr << "Failed to get full path for " << in_file << std::endl;
        return 1;
    }
    std::string full_in_file = full_path;
    // Detect number of CPUs
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    size_t num_cpus = sysinfo.dwNumberOfProcessors;
    std::vector<std::pair<size_t, std::vector<size_t>>> pred_list(pred_to_succ.begin(), pred_to_succ.end());
    const size_t window_size = std::min(pred_list.size(), num_cpus);  // Restore parallel processing
    std::cout << "Detected " << num_cpus << " CPUs, using window size " << window_size << std::endl;
    std::vector<HANDLE> processes;
    std::vector<HANDLE> child_stdout_reads;
    std::vector<HANDLE> child_stderr_reads;
    std::vector<size_t> current_preds;  // Track which pred each process is for
    size_t pred_index = 0;

    while (pred_index < pred_list.size() || !processes.empty()) {
        // Launch new processes up to window size
        while (processes.size() < window_size && pred_index < pred_list.size()) {
            const auto& pair = pred_list[pred_index];
            size_t pred = pair.first;
            const std::vector<size_t>& succ_list = pair.second;
            // Progress indicator
            double progress = (pred_index + 1.0) / pred_list.size() * 100.0;
            std::cout << "\rLaunching: " << std::fixed << std::setprecision(1) << progress 
                      << "% (" << (pred_index + 1) << "/" << pred_list.size() << " preds)" << std::flush;

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

            // Create pipes for child's stderr
            HANDLE hChildStderrRead, hChildStderrWrite;
            if (!CreatePipe(&hChildStderrRead, &hChildStderrWrite, &sa, 0)) {
                std::cerr << "Failed to create stderr pipe for pred " << pred << std::endl;
                CloseHandle(hChildStdinRead);
                CloseHandle(hChildStdinWrite);
                CloseHandle(hChildStdoutRead);
                CloseHandle(hChildStdoutWrite);
                return 1;
            }
            SetHandleInformation(hChildStderrRead, HANDLE_FLAG_INHERIT, 0);  // Parent reads, don't inherit
            SetHandleInformation(hChildStderrWrite, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);  // Child writes

            // Create process
            STARTUPINFO si = {sizeof(STARTUPINFO)};
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = hChildStdinRead;
            si.hStdOutput = hChildStdoutWrite;
            si.hStdError = hChildStderrWrite;

            PROCESS_INFORMATION pi;
            std::string cmd = std::string(exe_path) + " -oracle-child " + std::to_string(pred) + " " + std::to_string(file_data.size()) + " " + full_in_file + " " + in_file + ".segments";
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
            CloseHandle(hChildStderrWrite);

            // Write data to child's stdin
            DWORD written;
            WriteFile(hChildStdinWrite, &pred, sizeof(size_t), &written, NULL);

            uint64_t num_succ = succ_list.size();
            WriteFile(hChildStdinWrite, &num_succ, sizeof(uint64_t), &written, NULL);
            if (num_succ > 0) WriteFile(hChildStdinWrite, succ_list.data(), num_succ * sizeof(size_t), &written, NULL);

            // Close stdin write handle
            CloseHandle(hChildStdinWrite);

            processes.push_back(pi.hProcess);
            child_stdout_reads.push_back(hChildStdoutRead);
            child_stderr_reads.push_back(hChildStderrRead);
            current_preds.push_back(pred);

            //std::cout << "Launched process for pred = " << pred << std::endl;
            ++pred_index;
        }

        // Wait for one process to complete
        if (!processes.empty()) {
            DWORD wait_result = WaitForMultipleObjects(processes.size(), processes.data(), FALSE, INFINITE);
            if (wait_result >= WAIT_OBJECT_0 && wait_result < WAIT_OBJECT_0 + processes.size()) {
                size_t completed_index = wait_result - WAIT_OBJECT_0;
                HANDLE completed_process = processes[completed_index];
                HANDLE completed_stdout = child_stdout_reads[completed_index];
                HANDLE completed_stderr = child_stderr_reads[completed_index];

                // Get exit code
                DWORD exit_code;
                GetExitCodeProcess(completed_process, &exit_code);
                if (exit_code != 0) {
                    std::cerr << "Child process for pred " << current_preds[completed_index] << " failed with exit code " << exit_code << ", skipping" << std::endl;
                    // pred_costs[pred] remains empty
                } else {
                    // Channel separation: read results from stdout pipe
                    DWORD bytesRead;
                    uint64_t num_results;
                    if (!ReadFile(completed_stdout, &num_results, sizeof(uint64_t), &bytesRead, NULL) || bytesRead != sizeof(uint64_t)) {
                        std::cerr << "Failed to read num_results from pipe for pred " << current_preds[completed_index] << std::endl;
                        // pred_costs[pred] remains empty
                    } else {
                        //std::cout << "Read num_results: " << num_results << " for pred " << current_preds[completed_index] << std::endl;
                        for (uint64_t i = 0; i < num_results; ++i) {
                            size_t succ;
                            if (!ReadFile(completed_stdout, &succ, sizeof(size_t), &bytesRead, NULL) || bytesRead != sizeof(size_t)) {
                                std::cerr << "Failed to read succ from pipe" << std::endl;
                                break;
                            }
                            uint64_t num_costs;
                            if (!ReadFile(completed_stdout, &num_costs, sizeof(uint64_t), &bytesRead, NULL) || bytesRead != sizeof(uint64_t)) {
                                std::cerr << "Failed to read num_costs from pipe" << std::endl;
                                break;
                            }
                            for (uint64_t j = 0; j < num_costs; ++j) {
                                size_t pred_read;
                                double cost;
                                if (!ReadFile(completed_stdout, &pred_read, sizeof(size_t), &bytesRead, NULL) || bytesRead != sizeof(size_t)) {
                                    std::cerr << "Failed to read pred_read from pipe" << std::endl;
                                    break;
                                }
                                if (!ReadFile(completed_stdout, &cost, sizeof(double), &bytesRead, NULL) || bytesRead != sizeof(double)) {
                                    std::cerr << "Failed to read cost from pipe" << std::endl;
                                    break;
                                }
                                pred_costs[pred_read].emplace_back(succ, cost);
                            }
                        }
                    }
                }

                // Close handles
                CloseHandle(completed_process);
                CloseHandle(completed_stdout);
                CloseHandle(completed_stderr);

                // Remove from vectors
                processes.erase(processes.begin() + completed_index);
                child_stdout_reads.erase(child_stdout_reads.begin() + completed_index);
                child_stderr_reads.erase(child_stderr_reads.begin() + completed_index);
                current_preds.erase(current_preds.begin() + completed_index);
            }
        }
    }

    debugLog("run_oracle_multiprocess end");
    return 0;
}

int main(int argc, char* argv[]) {
  // if (!kReleaseBuild) {
  //   RunAllTests();
  // }
  Options options;
  auto ret = options.parse(argc, argv);
  if (ret) {
    std::cerr << "Failed to parse arguments" << std::endl;
    return ret;
  }
  switch (options.mode) {
  case Options::kModeOracleChild:
    return OracleChildMain(argc, argv);
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
    if (options.files.empty()) {
      options.usage(argv[0]);
      return 1;
    }
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
    try {
    printHeader();
    std::cout << "Running in Observer Mode" << std::endl;
    // Read the input file
    std::vector<FileInfo> files = options.files;
    uint64_t total_size = 0;
    for (const auto& f : files) {
      File fin;
      if (fin.open(f.getName(), std::ios_base::in | std::ios_base::binary) != 0) {
        std::cerr << "Error opening: " << f.getName() << " errno: " << errno << std::endl;
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
    cm::CM<8, false> compressor(FrequencyCounter<256>(), 8, false, Detector::kProfileSimple);
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
    } catch (const std::exception& e) {
      std::cerr << "Exception in observer: " << e.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Unknown exception in observer" << std::endl;
      return 1;
    }
    break;
  }
  case Options::kModeSegment: {
    printHeader();
    std::cout << "Running Auto-Tuned MDL-PELT Segmentation" << std::endl;

    // 1. Configuration
    const size_t BLOCK_SIZE = (options.segment_window > 0) ? options.segment_window : 512;
    std::string in_file = options.files[0].getName();
    
    // 2. Load Data & Build Prefix Histograms (Heavy Lift - Do Once)
    std::ifstream fin(in_file, std::ios::binary | std::ios::ate);
    if (!fin) { std::cerr << "Error: " << in_file << std::endl; return 1; }
    size_t file_size = fin.tellg();
    fin.seekg(0);
    size_t num_blocks = (file_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::cout << "File: " << formatNumber(file_size) << " bytes | Blocks: " << num_blocks << std::endl;
    std::cout << "Building Histograms..." << std::flush;

    std::vector<uint8_t> buffer(BLOCK_SIZE);
    // Use uint32_t to save RAM (4 bytes vs 8 bytes) 
    std::vector<std::array<uint32_t, 256>> prefix_counts;
    prefix_counts.reserve(num_blocks + 1);
    prefix_counts.push_back({}); // 0-th is empty
    std::fill(prefix_counts[0].begin(), prefix_counts[0].end(), 0);

    std::array<uint32_t, 256> current_counts = {0};
    for (size_t i = 0; i < num_blocks; ++i) {
        size_t read_size = std::min(BLOCK_SIZE, file_size - (i * BLOCK_SIZE));
        fin.read(reinterpret_cast<char*>(buffer.data()), read_size);
        for (size_t k = 0; k < read_size; ++k) current_counts[buffer[k]]++;
        prefix_counts.push_back(current_counts);
    }
    fin.close();
    std::cout << " Done." << std::endl;

    // 3. Pre-compute Log Table
    const size_t LOG_TABLE_SIZE = 65536; 
    std::vector<double> x_log_x(LOG_TABLE_SIZE);
    x_log_x[0] = 0.0;
    for (size_t i = 1; i < LOG_TABLE_SIZE; ++i) x_log_x[i] = i * std::log2(i);

    // 4. The PELT Solver (Reusable Lambda)
    const double log_N = std::log2(file_size);
    const double K = 255.0; 

    // Helper to calc cost between blocks
    auto GetSegCost = [&](size_t start, size_t end, double penalty) -> double {
        const auto& c_end = prefix_counts[end];
        const auto& c_start = prefix_counts[start];
        size_t len = (end == num_blocks) ? (file_size - start * BLOCK_SIZE) : ((end - start) * BLOCK_SIZE);
        if (len == 0) return 0.0;

        double sum_c_log_c = 0.0;
        for (int i = 0; i < 256; ++i) {
            uint32_t count = c_end[i] - c_start[i];
            if (count > 0) {
                if (count < LOG_TABLE_SIZE) sum_c_log_c += x_log_x[count];
                else sum_c_log_c += count * std::log2(count);
            }
        }
        double term_L_log_L = (len < LOG_TABLE_SIZE) ? x_log_x[len] : (len * std::log2(len));
        return (term_L_log_L - sum_c_log_c) + penalty;
    };

    auto RunPELT = [&](double factor) -> std::vector<size_t> {
        double penalty = 0.5 * K * log_N * factor;
        std::vector<double> F(num_blocks + 1, 1e18);
        std::vector<int> bp(num_blocks + 1, -1);
        std::vector<size_t> candidates = {0};
        F[0] = -penalty; 

        for (size_t t = 1; t <= num_blocks; ++t) {
            double min_cost = 1e18;
            int best_tau = -1;
            
            // Update
            for (size_t tau : candidates) {
                double c = F[tau] + GetSegCost(tau, t, penalty);
                if (c < min_cost) { min_cost = c; best_tau = tau; }
            }
            F[t] = min_cost;
            bp[t] = best_tau;

            // Prune
            auto it = std::remove_if(candidates.begin(), candidates.end(), [&](size_t tau) {
                return (F[tau] + GetSegCost(tau, t, penalty)) > F[t];
            });
            candidates.erase(it, candidates.end());
            candidates.push_back(t);
        }

        // Backtrack
        std::vector<size_t> cuts;
        size_t curr = num_blocks;
        while (curr > 0) {
            cuts.push_back(curr);
            curr = bp[curr];
        }
        cuts.push_back(0);
        std::reverse(cuts.begin(), cuts.end());
        return cuts;
    };

    // 5. Auto-Tuning Loop
    // Target: Avg segment size ~16KB (Adjustable)
    size_t target_count = file_size / 16384; 
    if (target_count < 50) target_count = 50; // Minimum sanity
    if (target_count > 50000) target_count = 50000; // Cap

    double low = 0.05, high = 5.0;
    double best_factor = 1.2;
    std::vector<size_t> best_cuts;
    
    // Or use user override if provided
    if (options.segment_threshold > 0.0) {
        std::cout << "Using manual threshold: " << options.segment_threshold << std::endl;
        best_cuts = RunPELT(options.segment_threshold);
    } else {
        std::cout << "Auto-Tuning for ~" << target_count << " segments..." << std::endl;
        for (int i = 0; i < 10; ++i) {
            double mid = (low + high) / 2.0;
            auto cuts = RunPELT(mid);
            size_t count = cuts.size() - 1;
            
            std::cout << "  Iter " << i << ": Factor=" << mid << " -> " << count << " segs" << std::endl;
            
            if (abs((long)count - (long)target_count) < (target_count * 0.1)) {
                best_cuts = std::move(cuts);
                best_factor = mid;
                break; // Close enough
            }
            
            if (count < target_count) high = mid; // Too few -> Lower penalty
            else low = mid; // Too many -> Raise penalty
            
            if (i == 9) best_cuts = std::move(cuts); // Take last result
        }
    }

    std::cout << "Final: " << (best_cuts.size() - 1) << " segments." << std::endl;

    // Convert to byte boundaries
    std::vector<size_t> boundaries;
    boundaries.push_back(0);
    for (size_t block : best_cuts) {
      if (block > 0) {
        boundaries.push_back(block * BLOCK_SIZE);
      }
    }
    boundaries.push_back(file_size);

    // Boundary Refinement Phase ("Inductive Boundary Snap")
    std::cout << "Refining " << boundaries.size() << " boundaries..." << std::endl;
    std::string original_file = in_file;
    std::ifstream fin_refine(original_file, std::ios::binary);
    if (fin_refine) {
      const size_t SEARCH_RADIUS = BLOCK_SIZE / 2;
      for (size_t i = 1; i < boundaries.size() - 1; ++i) {
        size_t coarse_cut = boundaries[i];
        size_t search_start = (coarse_cut > SEARCH_RADIUS) ? coarse_cut - SEARCH_RADIUS : 0;
        size_t search_end = std::min(file_size, coarse_cut + SEARCH_RADIUS);
        
        // Constrain to neighbors to preserve segment order
        if (search_start <= boundaries[i - 1]) search_start = boundaries[i - 1] + 1;
        if (search_end >= boundaries[i + 1]) search_end = boundaries[i + 1] - 1;
        
        size_t window_len = search_end - search_start;
        if (window_len < 2) continue;
        std::vector<uint8_t> window(window_len);
        fin_refine.seekg(search_start);
        fin_refine.read(reinterpret_cast<char*>(window.data()), window_len);

        // Initialize Histograms
        std::array<uint32_t, 256> right_counts = {0};
        std::array<uint32_t, 256> left_counts = {0};
        double right_sum = 0.0;
        double left_sum = 0.0;

        // Start with everything in the Right partition
        for (uint8_t b : window) {
          if (right_counts[b] > 0) right_sum -= x_log_x[right_counts[b]];
          right_counts[b]++;
          right_sum += x_log_x[right_counts[b]];
        }

        double min_local_cost = std::numeric_limits<double>::infinity();
        size_t best_local_offset = 0;

        // Sweep the cut point from Left to Right
        for (size_t k = 0; k < window_len - 1; ++k) {
          uint8_t b = window[k];
          
          // Move byte 'b' from Right to Left
          right_sum -= x_log_x[right_counts[b]];
          right_counts[b]--;
          if (right_counts[b] > 0) right_sum += x_log_x[right_counts[b]];

          if (left_counts[b] > 0) left_sum -= x_log_x[left_counts[b]];
          left_counts[b]++;
          left_sum += x_log_x[left_counts[b]];

          size_t len_l = k + 1;
          size_t len_r = window_len - len_l;
          
          // Calculate Entropy Cost: L*log(L) - sum(c*log(c))
          double cost_l = x_log_x[len_l] - left_sum;
          double cost_r = x_log_x[len_r] - right_sum;
          double total = cost_l + cost_r;

          if (total < min_local_cost) {
            min_local_cost = total;
            best_local_offset = k + 1;
          }
        }
        
        // Update to the optimal byte-perfect boundary
        boundaries[i] = search_start + best_local_offset;
      }
      fin_refine.close();
      std::cout << "Refinement Complete." << std::endl;
    } else {
      std::cerr << "Warning: Could not reopen file for refinement" << std::endl;
    }

    // 6. Write Output
    std::string out_file = in_file + ".segments";
    std::ofstream ofs(out_file);
    if (!ofs) return 1;
    ofs << (boundaries.size() - 1) << std::endl;
    for (size_t i = 0; i < boundaries.size() - 1; ++i) {
        size_t start = boundaries[i];
        size_t len = boundaries[i+1] - boundaries[i];
        ofs << start << " " << len << std::endl;
    }
    ofs.close();
    std::cout << "Wrote segments to " << out_file << std::endl;
    break;
  }
  case Options::kModeFingerprint: {
    printHeader();
    std::cout << "Running Hybrid Holographic Fingerprinting Mode" << std::endl;
    // Read .segments file
    if (argc != 3) {
      std::cerr << "Usage: mcm -fingerprint <original_file>" << std::endl;
      return 1;
    }
    std::string in_file = argv[2];
    std::cout << "in_file: " << in_file << std::endl;
    std::ifstream ifs(in_file + ".segments");
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
    std::string original_file = in_file;

    // Load original file
    File fin;
    if (fin.open(original_file, std::ios_base::in | std::ios_base::binary)) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    std::vector<uint8_t> file_data(fin.length());
    fin.read(&file_data[0], fin.length());
    fin.close();

    // Filter valid segments
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

    // Compute Hybrid Fingerprints
    std::vector<Fingerprint> fingerprints(num_valid);
    std::cout << "Computing Hybrid Holographic Fingerprints..." << std::endl;

    // #pragma omp parallel for
    for (int i = 0; i < (int)num_valid; ++i) {
      size_t start = valid_segments[i].first;
      size_t len = valid_segments[i].second;
      // Clamp len for feature extraction speed (first 16KB is usually enough for signature)
      size_t scan_len = std::min(len, (size_t)16384);
      std::vector<uint8_t> segment_data(file_data.begin() + start, file_data.begin() + start + scan_len);
      fingerprints[i] = compute_fingerprint(segment_data);
      if (i % 100 == 0) std::cout << "\rScan: " << i << "/" << num_valid << std::flush;
    }
    std::cout << std::endl;

    // Matching Loop (Asymmetric)
    size_t top_k = options.fingerprint_top_k;
    std::vector<std::vector<size_t>> candidates(num_valid);

    // #pragma omp parallel for
    for (int i = 0; i < (int)num_valid; ++i) {
      std::vector<std::pair<float, size_t>> distances;
      distances.reserve(num_valid);

      for (size_t j = 0; j < num_valid; ++j) {
        if (i == j) continue;

        // Note: Distance(Donor=j, Recv=i)
        // We are looking for the best PREDECESSOR (j) for current segment (i)
        float dist = hybrid_distance(fingerprints[j], fingerprints[i]);
        distances.emplace_back(dist, j);
      }

      if (distances.size() > top_k) {
          std::partial_sort(distances.begin(), distances.begin() + top_k, distances.end());
          distances.resize(top_k);
      } else {
          std::sort(distances.begin(), distances.end());
      }

      for (const auto& pair : distances) {
        candidates[i].push_back(pair.second);
      }

      if (i % 10 == 0) std::cout << "\rMatch: " << i << "/" << num_valid << std::flush;
    }
    std::cout << std::endl;

    // Output .candidates file
    std::string out_file = in_file + ".segments.candidates";
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

    // Save fingerprint data for reuse in oracle mode
    std::string fingerprint_file = in_file + ".segments.fingerprints";
    WriteBinary(fingerprint_file, fingerprints);
    std::cout << "Saved " << num_valid << " fingerprint structures to " << fingerprint_file << std::endl;
    break;
  }
  {
  case Options::kModeOracle: {
    printHeader();
    std::cout << "Running Oracle Mode" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
      std::cerr << "Usage: mcm -oracle <original_file>" << std::endl;
      return 1;
    }
    std::string in_file = argv[2];
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
      size_t i = std::stoul(token);
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
            cands.push_back(std::stoul(token));
          }
        }
      }
    }
    size_t num_segments = candidates.size();
    std::cout << "Read " << num_segments << " segments from candidates" << std::endl;
    if (candidates.size() > 0) std::cout << "candidates[0].size() = " << candidates[0].size() << std::endl;

    // Determine segments file
    std::string segments_file = in_file + ".segments";
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
    if (file_size > 0 && !fin.read((char*)&file_data[0], file_size)) {
      std::cerr << "Error reading original file: " << original_file << std::endl;
      return 1;
    }
    fin.close();

    // Filter valid segments as in fingerprint
    std::vector<std::pair<size_t, size_t>> valid_segments = segments;

    // Build reverse map: pred -> list of succ that have pred as candidate
    std::map<size_t, std::vector<size_t>> pred_to_succ;
    for (size_t pred = 0; pred < candidates.size(); ++pred) {
      if (!candidates[pred].empty()) {
        for (auto it = candidates[pred].begin(); it != candidates[pred].end(); ) {
          if (*it >= num_segments) {
            it = candidates[pred].erase(it);
          } else {
            ++it;
          }
        }
        if (!candidates[pred].empty()) {
          pred_to_succ[pred] = candidates[pred];
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
    //     if (std::find(succs.begin(), succs.end(), 0) != succs.end()) {
    //         test_pred_to_succ[1] = {0};
    //     }
    // }
    // pred_to_succ = test_pred_to_succ;

    // For each pred, compress pred once, take snapshot, then evaluate all succ that have pred as candidate
    std::vector<std::vector<std::pair<size_t, double>>> pred_costs(num_segments);  // for each pred, list of (succ, cost)
    std::cout << "Number of pred to process: " << pred_to_succ.size() << std::endl << std::flush;

    // 1. Pre-compute Alone Costs
    std::cout << "Pre-computing Alone Costs: 0.0% (0/" << num_segments << " segments)" << std::flush;
    std::vector<double> global_alone_costs(num_segments);
    // Use serial for now
    for (size_t i = 0; i < num_segments; ++i) {
        // Get segment data
        size_t start = valid_segments[i].first;
        size_t len = valid_segments[i].second;
        size_t head_len = std::min((size_t)10240, len);
        std::vector<uint8_t> head_data(head_len);
        memcpy(head_data.data(), file_data.data() + start, head_len);

        // Compress alone
        cm::CM<8, false> cm_alone(FrequencyCounter<256>(), 0, false, Detector::kProfileSimple);
        cm_alone.cur_profile_ = cm::CMProfile();
        for (int j = 0; j < static_cast<int>(cm::kModelCount); ++j) {
            cm_alone.cur_profile_.EnableModel(static_cast<cm::ModelType>(j));
        }
        cm_alone.cur_profile_.SetMatchModelOrder(12);
        cm_alone.cur_profile_.SetMinLZPLen(10);
        cm_alone.observer_mode = true;
        cm_alone.init();
        MemoryReadStream in_alone(head_data);
        VoidWriteStream out_alone;
        cm_alone.compress(&in_alone, &out_alone, head_data.size());
        double alone_bits = 0.0;
        for (double e : cm_alone.entropies) alone_bits += e;
        global_alone_costs[i] = alone_bits;

        // Progress
        double percent = 100.0 * (i + 1) / num_segments;
        std::cout << "\rPre-computing Alone Costs: " << std::fixed << std::setprecision(1) << percent << "% (" << (i + 1) << "/" << num_segments << " segments)" << std::flush;
    }
    std::cout << std::endl;
    std::cout << "Pre-computed alone costs for " << num_segments << " segments" << std::endl;

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
    const int BATCH_SIZE = 10;

    auto worker_func = [&]() {
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
                WriteFile(hChildInWrite, &pred_id, sizeof(size_t), &written, NULL);
                uint64_t num_succs = succs.size();
                WriteFile(hChildInWrite, &num_succs, sizeof(uint64_t), &written, NULL);
                if (!succs.empty()) WriteFile(hChildInWrite, succs.data(), sizeof(size_t) * succs.size(), &written, NULL);
                if (!succ_alone_costs.empty()) WriteFile(hChildInWrite, succ_alone_costs.data(), sizeof(double) * succ_alone_costs.size(), &written, NULL);
                FlushFileBuffers(hChildInWrite);

                // Read from Child
                uint64_t num_results;
                ReadFile(hChildOutRead, &num_results, sizeof(uint64_t), &written, NULL);
                for (uint64_t r = 0; r < num_results; ++r) {
                    size_t succ;
                    ReadFile(hChildOutRead, &succ, sizeof(size_t), &written, NULL);
                    uint64_t num_costs;
                    ReadFile(hChildOutRead, &num_costs, sizeof(uint64_t), &written, NULL);
                    for (uint64_t c = 0; c < num_costs; ++c) {
                        size_t p;
                        double cost;
                        ReadFile(hChildOutRead, &p, sizeof(size_t), &written, NULL);
                        ReadFile(hChildOutRead, &cost, sizeof(double), &written, NULL);
                        std::lock_guard<std::mutex> lock(results_mutex);
                        pred_costs[p].emplace_back(succ, cost);
                    }
                }
                processed_in_batch++;
            }

            processed_preds += processed_in_batch;
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
    if (num_cpus == 0) num_cpus = 4;
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
    break;
  }
  case Options::kModePathCover: {
    printHeader();
    std::cout << "Running PathCover Mode" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
      std::cerr << "Usage: mcm -pathcover <original_file>" << std::endl;
      return 1;
    }
    std::string in_file = std::string(argv[2]) + ".segments";
    std::cout << "in_file: " << in_file << std::endl;
    // Read segments file
    std::string segments_file = in_file;
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

    // Read oracle - NEW FORMAT: multiple candidates per predecessor
    struct Edge {
      size_t to;
      double cost;
    };
    std::vector<std::vector<Edge>> candidates(num_segments);
    std::ifstream oracle_ifs(original_file + ".oracle");
    if (!oracle_ifs) {
      std::cerr << "Error opening oracle file: " << in_file << std::endl;
      return 1;
    }
    std::string line;
    while (std::getline(oracle_ifs, line)) {
      std::istringstream iss(line);
      std::string token;
      std::getline(iss, token, ':');
      size_t pred_id = std::stoul(token);
      std::string transitions_str;
      std::getline(iss, transitions_str);
      std::istringstream trans_iss(transitions_str);
      std::string trans_token;
      while (std::getline(trans_iss, trans_token, ';')) {
        if (trans_token.empty()) continue;
        std::istringstream pair_iss(trans_token);
        std::string succ_str, cost_str;
        std::getline(pair_iss, succ_str, ',');
        std::getline(pair_iss, cost_str);
        size_t succ_id = std::stoul(succ_str);
        double cost = std::stod(cost_str);
        candidates[pred_id].push_back({succ_id, cost});
      }
      // Sort by cost ascending (Stable)
      std::sort(candidates[pred_id].begin(), candidates[pred_id].end(), 
                [](const Edge& a, const Edge& b) { 
                    if (std::abs(a.cost - b.cost) > 1e-9) return a.cost < b.cost;
                    return a.to < b.to; // Break ties with successor ID
                });

      // RED TEAM FIX: Protect Natural Predecessor from eviction
      // If we resize to 32, we MUST ensure the natural edge (pred_id + 1) stays.
      if (candidates[pred_id].size() > 32) {
        size_t natural_succ = pred_id + 1;
        bool natural_found = false;
        size_t natural_idx = -1;
        
        // Check if natural successor is in the list
        for (size_t k = 0; k < candidates[pred_id].size(); ++k) {
            if (candidates[pred_id][k].to == natural_succ) {
                natural_found = true;
                natural_idx = k;
                break;
            }
        }

        // Resize to 31 to leave room if we need to force-keep natural
        // or resize to 32 if natural is already safe in top 32
        if (natural_found && natural_idx < 32) {
             candidates[pred_id].resize(32);
        } else if (natural_found && natural_idx >= 32) {
             // Natural was found but is about to be cut. Save it.
             Edge natural_edge = candidates[pred_id][natural_idx];
             candidates[pred_id].resize(31);
             candidates[pred_id].push_back(natural_edge); // Put it back
        } else {
             candidates[pred_id].resize(32); // Natural not in list at all
        }
      }
    }
    oracle_ifs.close();
    size_t total_candidates = 0;
    for (const auto& c : candidates) total_candidates += c.size();
    std::cout << "Read " << total_candidates << " candidate transitions from oracle" << std::endl;

    // --- STEP 1: Calculate Donor Scores ---
    // A negative cost means the predecessor makes the successor smaller (Savings).
    // We want to find "Universal Donors" (High negative sum).
    std::vector<double> donor_scores(num_segments, 0.0);
    for (size_t i = 0; i < num_segments; ++i) {
        for (const auto& edge : candidates[i]) {
            // Only count "savings" (negative costs), ignore "pollution" (positive costs)
            if (edge.cost < 0) {
                donor_scores[i] += edge.cost;
            }
        }
    }
    
    // Beam Search Parameters
    const size_t BEAM_WIDTH = 2000;
    const double ORPHAN_PENALTY = 1000.0;
    
    // RED TEAM FIX: Threshold is in TOTAL BITS. 
    // 2.0 was too small. We require ~64 bytes (512 bits) of savings to justify a jump.
    const double FUSION_THRESHOLD = 512.0;
    
    // Debug: Print top donors
    std::vector<std::pair<double, size_t>> top_donors;
    for (size_t i = 0; i < num_segments; ++i) top_donors.push_back({donor_scores[i], i});
    std::sort(top_donors.begin(), top_donors.end()); // Ascending (most negative first)
    if (top_donors.size() >= 3) {
        std::cout << "Top 3 Donors: " 
                  << top_donors[0].second << " (" << top_donors[0].first << "), "
                  << top_donors[1].second << " (" << top_donors[1].first << "), "
                  << top_donors[2].second << " (" << top_donors[2].first << ")" << std::endl;
    }

    // --- GRAPH OPTIMIZATION: Pre-compute allowed fusions with adjusted costs ---
    std::vector<std::vector<Edge>> allowed_fusions(num_segments);
    for (size_t pred = 0; pred < num_segments; ++pred) {
      for (const auto& edge : candidates[pred]) {
        size_t succ = edge.to;
        double cost = edge.cost;
        bool is_natural = (succ == pred + 1);
        double adjusted_cost = cost;
        
        if (is_natural) {
          // Apply incumbency bonus
          double bonus = (cost < 0) ? (cost * 0.10) : -500.0;
          adjusted_cost += bonus;
        } else {
          // For non-natural, only allow if significant savings
          if (cost > -FUSION_THRESHOLD) continue;
        }
        
        allowed_fusions[pred].push_back({succ, adjusted_cost});
      }
    }
    std::cout << "Pre-computed allowed fusions for graph optimization" << std::endl;

    // --- GGCA-IR Algorithm ---

    // Data Structures
    struct UnionFind {
      std::vector<int> parent;
      UnionFind(size_t n) {
        parent.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
      }
      int find(int v) {
        if (v == parent[v]) return v;
        return parent[v] = find(parent[v]);
      }
      void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[b] = a;
      }
    };

    struct EdgeSavings {
      uint32_t u, v;
      double savings;
    };

    std::vector<int> next_seg(num_segments, -1);
    std::vector<int> prev_seg(num_segments, -1);
    UnionFind dsu(num_segments);

    // Phase 1: Global Edge Collection & Sorting
    const double RESET_PENALTY = 2000.0;
    std::vector<EdgeSavings> all_edges;
    for (size_t pred = 0; pred < num_segments; ++pred) {
      for (const auto& edge : allowed_fusions[pred]) { // <--- FIX: Use filtered list
        size_t succ = edge.to;
        double cost = edge.cost;
        double savings = RESET_PENALTY - cost;
        bool is_natural = (succ == pred + 1);
        if (is_natural) {
          savings += 0.1 * std::abs(cost);
        }
        if (savings > 0) {
          all_edges.push_back({static_cast<uint32_t>(pred), static_cast<uint32_t>(succ), savings});
        }
      }
    }
    std::sort(all_edges.begin(), all_edges.end(), [](const EdgeSavings& a, const EdgeSavings& b) {
      return a.savings > b.savings; // Descending
    });
    std::cout << "Collected " << all_edges.size() << " edges for GGCA-IR" << std::endl;

    // Phase 2: Greedy Construction (Kruskal Pass)
    for (const auto& edge : all_edges) {
      uint32_t u = edge.u;
      uint32_t v = edge.v;
      if (next_seg[u] == -1 && prev_seg[v] == -1 && dsu.find(u) != dsu.find(v)) {
        next_seg[u] = v;
        prev_seg[v] = u;
        dsu.unite(u, v);
      }
    }
    std::cout << "Completed Greedy Construction" << std::endl;

    // Phase 3 & 4: Deep Iterative Refinement
    // Run until convergence or max passes
    bool improved = true;
    int pass = 0;
    const int MAX_PASSES = 20; 

    while (improved && pass < MAX_PASSES) {
        improved = false;
        pass++;
        size_t changes = 0;

        for (size_t u = 0; u < num_segments; ++u) {
            for (const auto& edge : allowed_fusions[u]) { // <--- FIX: Use filtered list
                size_t v = edge.to;
                double cost = edge.cost;

                // 1. STANDARD STITCH (Tail -> Head)
                if (next_seg[u] == -1 && prev_seg[v] == -1) {
                    if (dsu.find(u) != dsu.find(v)) {
                         if (cost < RESET_PENALTY) {
                             next_seg[u] = v; prev_seg[v] = u; dsu.unite(u, v);
                             improved = true; changes++;
                         }
                    }
                }
                // 2. AGGRESSIVE STEAL (Target Busy)
                // We want u -> v, but v has old_pred.
                // Condition: u is Tail (free), v is Busy.
                else if (next_seg[u] == -1 && prev_seg[v] != -1) {
                    int old_pred = prev_seg[v];
                    // Find cost of old_pred -> v
                    double old_cost = 1e9;
                    for(auto& e : candidates[old_pred]) if(e.to == v) { old_cost = e.cost; break; }
                    
                    if (dsu.find(u) != dsu.find(v) && cost < old_cost) {
                        next_seg[old_pred] = -1; // Break old
                        next_seg[u] = v; prev_seg[v] = u; // Link new
                        improved = true; changes++;
                    }
                }
                // 3. AGGRESSIVE BREAK (Source Busy)
                // We want u -> v, but u has old_next.
                // Condition: u is Busy, v is Head (free).
                else if (next_seg[u] != -1 && prev_seg[v] == -1) {
                    int old_next = next_seg[u];
                    if (dsu.find(u) != dsu.find(v)) {
                         double old_cost = 1e9;
                         for(auto& e : candidates[u]) if(e.to == old_next) { old_cost = e.cost; break; }

                         if (cost < old_cost) {
                             prev_seg[old_next] = -1; // Break old
                             next_seg[u] = v; prev_seg[v] = u; // Link new
                             improved = true; changes++;
                         }
                    }
                }
            }
        }

        // Rebuild DSU if we made structural changes (links broken)
        if (improved) {
            dsu = UnionFind(num_segments);
            for(size_t i=0; i<num_segments; ++i) {
                if (next_seg[i] != -1) dsu.unite(i, next_seg[i]);
            }
            std::cout << "Pass " << pass << ": " << changes << " optimizations applied." << std::endl;
        }
    }

    // --- Output Generation ---
    // Build chains from next_seg
    std::vector<std::vector<size_t>> chains;
    std::vector<bool> visited(num_segments, false);
    for (size_t i = 0; i < num_segments; ++i) {
      if (!visited[i] && prev_seg[i] == -1) {
        // Start of a chain
        std::vector<size_t> chain;
        size_t current = i;
        while (current != static_cast<size_t>(-1)) {
          visited[current] = true;
          chain.push_back(current);
          current = next_seg[current];
        }
        chains.push_back(chain);
      }
    }

    // Collect orphans
    std::vector<size_t> orphans;
    for (size_t i = 0; i < num_segments; ++i) {
      if (!visited[i]) {
        orphans.push_back(i);
      }
    }

    std::cout << "Built " << chains.size() << " chains and found " << orphans.size() << " orphans" << std::endl;

    // Handle orphans: sort by entropy cost ascending
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

    // --- STEP 2: Sort Chains by Cleanliness & Priority ---
    std::cout << "Sorting chains by Toxicity and Donor Score..." << std::endl;
    std::sort(chains.begin(), chains.end(), [&](const std::vector<size_t>& a, const std::vector<size_t>& b) {
        // 1. Calculate Toxicity (Average Entropy)
        auto chain_entropy = [&](const std::vector<size_t>& chain) {
            double total = 0.0;
            for (size_t s : chain) total += get_cost(s);
            // If get_cost returns total bits, divide by length. 
            // Here get_cost returns entropy*len or just len. 
            // We need Bits-Per-Byte approximation. 
            // Assuming get_cost approx equals compressed size in bits:
            double len = 0.0;
            for (size_t s : chain) len += segments[s].second;
            return (len > 0) ? (total / len) : 8.0; 
        };

        // Note: get_cost currently returns "Approximate Bits". 
        // We need to normalize if we want true "Bits Per Byte".
        // However, for sorting, "Total Entropy" (Cost) is a reasonable proxy for difficulty 
        // if segments are roughly same size.
        
        // Strategy: "Toxic Deferral"
        // If a chain is "Hard" (High Entropy), push it to the end.
        // We use a simplified check: Compare average per-segment cost.
        double cost_a = 0.0; for(auto s : a) cost_a += get_cost(s);
        double cost_b = 0.0; for(auto s : b) cost_b += get_cost(s);
        double avg_a = cost_a / a.size();
        double avg_b = cost_b / b.size();

        // Heuristic: If one is significantly "Harder" (> 20% diff), sort by difficulty (Easy First)
        // "Easy First" means Low Cost First.
        if (std::abs(avg_a - avg_b) > (std::min(avg_a, avg_b) * 0.2)) {
            return avg_a < avg_b;
        }

        // 2. Otherwise (similar difficulty), sort by Donor Score (Helpfulness)
        double score_a = 0.0; for (size_t seg : a) score_a += donor_scores[seg];
        double score_b = 0.0; for (size_t seg : b) score_b += donor_scores[seg];
        return score_a > score_b; // Descending (Best donors first)
    });
    std::cout << "Sorted " << chains.size() << " chains." << std::endl;
    // --- STEP 3: Smart Orphan Sort ---
    std::sort(orphans.begin(), orphans.end(), [&](size_t a, size_t b) {
        if (std::abs(donor_scores[a] - donor_scores[b]) > 1e-4) {
             return donor_scores[a] > donor_scores[b]; // Stronger donor first
        }
        return get_cost(a) < get_cost(b);
    });
    std::cout << "Sorted orphans by Donor Score -> Entropy." << std::endl;

    // Build reordered segment list: chains first (in some order), then orphans
    std::vector<size_t> reordered_segments;
    for (const auto& chain : chains) {
      reordered_segments.insert(reordered_segments.end(), chain.begin(), chain.end());
    }
    reordered_segments.insert(reordered_segments.end(), orphans.begin(), orphans.end());

    // Load original file data
    std::ifstream data_ifs(original_file, std::ios::binary | std::ios::ate);
    if (!data_ifs) {
      std::cerr << "Error opening original file: " << original_file << std::endl;
      return 1;
    }
    size_t file_size = data_ifs.tellg();
    data_ifs.seekg(0);
    std::vector<uint8_t> file_data(file_size);
    if (file_size > 0 && !data_ifs.read(reinterpret_cast<char*>(&file_data[0]), file_size)) {
      std::cerr << "Error reading original file: " << original_file << std::endl;
      return 1;
    }
    data_ifs.close();
    std::cout << "Loaded " << file_size << " bytes from original file" << std::endl;

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
      size_t current_start = reordered_segments[0]; // Renamed for clarity
      size_t count = 1;
      for (size_t i = 1; i < reordered_segments.size(); ++i) {
        // CHECK SEQUENTIALITY: Is this segment exactly 1 greater than the previous?
        if (reordered_segments[i] == current_start + count) {
          count++;
        } else {
          iofs << current_start << "," << count << std::endl;
          current_start = reordered_segments[i];
          count = 1;
        }
      }
      iofs << current_start << "," << count << std::endl;
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