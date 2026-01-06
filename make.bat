@echo off
echo Compiling...
"C:\temp_mingw\mingw64\bin\g++.exe" -pthread -DNDEBUG -O3 -fomit-frame-pointer -std=c++14 -Wno-deprecated-declarations -o mcm Archive.cpp Huffman.cpp MCM.cpp Memory.cpp Util.cpp Compressor.cpp File.cpp LZ.cpp Tests.cpp
if errorlevel 1 (
  echo Compilation failed.
) else (
  echo mcm.exe created.
)