#!/usr/bin/env python3
import struct

with open('testFiles/test_boundary.txt.entropy', 'rb') as f:
    data = f.read()

entropies = []
for i in range(0, len(data), 8):
    entropies.append(struct.unpack('<d', data[i:i+8])[0])

print(f"Total entropies: {len(entropies)}")
print("First 10:", entropies[:10])
print("Around 1000:", entropies[995:1005])
print("Around 4000:", entropies[3995:4005])
print("Around 5000:", entropies[4995:5005])
print("Max:", max(entropies))
print("Min:", min(entropies))