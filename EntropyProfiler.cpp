#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cerrno>
#include "File.hpp"
#include "CM.hpp"
#include "CM-inl.hpp"
#include "CCRConfig.h"
#include "Util.hpp"
#include "Archive.hpp"
#include "ProgressMeter.hpp"

int runEntropyProfiler(const std::vector<FileInfo>& files, const FileInfo& archive_file) {
    try {
        std::cout << "Profiling Entropy in Observer Mode (CCR Aligned)" << std::endl;

        // Read the input file
        std::vector<FileInfo> files_copy = files;
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

        // [FIX 1] INCREASE TEMPLATE SIZE & FORCE TEXT MODE
        // Changed <8, false> to <16, false> to fit 13 models.
        // Changed kProfileSimple to kProfileText to load text-specific structures.
        cm::CM<16, false> compressor(FrequencyCounter<256>(), 8, false, Detector::kProfileText);

        compressor.observer_mode = true;

        // [FIX 2] MANUAL INIT & OVERWRITE SEQUENCE
        // 1. Run standard initialization (allocates memory, tables, etc.)
        compressor.init();

        // 2. INJECT ECONOMY PROFILE
        // Overwrite the internal profile directly (it is public in CM.hpp)
        compressor.text_profile_ = GetCCRProfile();

        // 3. APPLY CHANGE
        // Tell the compressor to switch to Text Mode using the profile we just injected.
        // We use the scope operator :: to access the Enum.
        compressor.SetDataProfile(cm::CM<16, false>::kProfileText);

        // 4. DISABLE AUTO-INIT
        // Prevent compress() from running init() again and wiping our changes.
        compressor.skip_init = true;

        // Create streams
        ReadMemoryStream rms(buffer.data(), buffer.data() + buffer.size());
        VoidWriteStream vws;

        std::cout << "Starting aligned compress..." << std::endl;
        std::cout << "Compressing " << buffer.size() << " bytes" << std::endl;

        compressor.compress(&rms, &vws, buffer.size());

        std::cout << "entropies.size() = " << compressor.entropies.size() << std::endl;
        std::cout << "Compress done" << std::endl;

        // Skip stock compression size for observer to avoid hang
        uint64_t stock_size = 0;

        // Output entropies
        std::string out_file = archive_file.getName();
        if (out_file.empty()) {
            out_file = files_copy[0].getName() + ".entropy";
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
    return 0;
}