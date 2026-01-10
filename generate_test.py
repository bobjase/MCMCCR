#!/usr/bin/env python3
import os
import random

# Generate test file with alternating low and high CM entropy sections
english = 'a' * 10000  # Low entropy (highly predictable)
numbers = ''.join(str(random.randint(0,9)) for _ in range(10000))  # High entropy (unpredictable)
mandarin = chr(0x4E00) * 10000  # Low entropy
hebrew = ''.join(chr(random.randint(0x05D0, 0x05EA)) for _ in range(10000))  # High entropy

content = english + numbers + mandarin + hebrew

with open('testFiles/test_boundary.txt', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created test_boundary.txt with {len(content)} characters")