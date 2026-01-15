@echo off
echo Compiling...
"C:\temp_mingw\mingw64\bin\g++.exe" -static-libstdc++ -static-libgcc -msse4.2 -pthread -fopenmp -DNDEBUG -O3 -fomit-frame-pointer -std=c++14 -Wno-deprecated-declarations -o mcm.exe Archive.cpp Huffman.cpp MCM.cpp Memory.cpp Util.cpp Compressor.cpp File.cpp LZ.cpp Tests.cpp PathCover.cpp
if errorlevel 1 (
  echo Compilation failed.
) else (
  echo mcm.exe created.
)