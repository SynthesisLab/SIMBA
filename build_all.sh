#!/bin/bash

echo "Building VFB..."
cd vfb && make && cd ..
cp vfb/build/full-cache ./vfb_exe
echo "Done!"

echo "Building SIMBA..."
cd simba && make && cd ..
cp simba/simba_nocache ./simba_exe
cp simba/simba_rtid ./simba_rtid_exe
echo "Done!"

echo "Generating formulas..."
cd scripts
uv run generate.py -s 5 30 -v 2 10 -r 10 -o ../formulas.txt --seed 42
echo "Done!"
echo "Generating instances..."
uv run sample.py -i ../formulas.txt -n 16 -o ../instances/ --seed 42
cd ..
echo "Done!"