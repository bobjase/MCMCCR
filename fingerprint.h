#ifndef FINGERPRINT_H
#define FINGERPRINT_H

#include <string>
#include <vector>
#include <cstdint>

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

Fingerprint compute_fingerprint(const std::vector<uint8_t>& segment);
float hybrid_distance(const Fingerprint& donor, const Fingerprint& recv);

void runFingerprint(const std::string& original_file, int top_k);

#endif // FINGERPRINT_H