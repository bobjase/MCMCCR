#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <ctime>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <io.h>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <queue>
#include <sstream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <thread>
#include <vector>
#include <windows.h>
#include <eh.h>

#include "Archive.hpp"
#include "CCRConfig.h"
#include "CM.hpp"
#include "CM-inl.hpp"
#include "DeltaFilter.hpp"
#include "Dict.hpp"
#include "EntropySegmenter.h"
#include "File.hpp"
#include "fingerprint.h"
#include "Huffman.hpp"
#include "LZ-inl.hpp"
#include "OracleChild.h"
#include "PathCover.h"
#include "ProgressMeter.hpp"
#include "Segmenter.h"
#include "Tests.hpp"
#include "TurboCM.hpp"
#include "Util.hpp"
#include "X86Binary.hpp"

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
  size_t segment_window = 192;
  float segment_threshold = 0.0f;
  size_t segment_min_segment = 4096;
  size_t segment_max_segment = 65536;  // 64KB max segment size
  size_t segment_lookback = 1024;  // Fingerprinting parameters
  size_t fingerprint_top_k = 2048;
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

int runEntropyProfiler(const std::vector<FileInfo>& files, const FileInfo& archive_file);

int runOracle(const std::string& in_file);

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
    // not used in our code
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
    return runEntropyProfiler(options.files, options.archive_file);
  }
  case Options::kModeSegment: {
    if (options.files.empty()) { std::cerr << "Error: No input file." << std::endl; return 1; }
    std::string in_file = options.files[0].getName();
    runSegmenter(in_file, options);
    break;
  }
  case Options::kModeFingerprint: {
    if (argc != 3) {
      std::cerr << "Usage: mcm -fingerprint <original_file>" << std::endl;
      return 1;
    }
    std::string original_file = argv[2];
    runFingerprint(original_file, options.fingerprint_top_k);
    break;
  }
  case Options::kModeOracle: {
    if (argc != 3) {
      std::cerr << "Usage: mcm -oracle <original_file>" << std::endl;
      return 1;
    }
    return runOracle(argv[2]);
  }
  case Options::kModePathCover: {
    if (argc != 3) {
      std::cerr << "Usage: mcm -pathcover <original_file>" << std::endl;
      return 1;
    }
    std::string original_file = std::string(argv[2]);
    runPathCover(original_file);
    break;
  }
  }
  return 0;
}
