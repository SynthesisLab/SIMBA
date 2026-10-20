#!/bin/bash
set -e

# Runs that time out (or fail) are skipped and produce no line in the results.

TIMEOUT_S=60

echo "Running CVC5 on all instances..."
echo "instance,formula,time" > res_cvc5.csv
cd cvc5
uv sync
for i in ../instances/*.json
do
   timeout ${TIMEOUT_S} uv run --no-sync main.py $i >> ../res_cvc5.csv || true
done
cd ..
echo "Done!"


echo "Running SIMBA on all instances..."
echo "instance,formula,time" > res_simba.csv
for i in ./instances/*.json
do
   timeout ${TIMEOUT_S} ./simba_exe $i 20 >> res_simba.csv || true
done
echo "Done!"


echo "Running SIMBA-RTid on all instances..."
echo "instance,formula,time" > res_simba_rtid.csv
for i in ./instances/*.json
do
   timeout ${TIMEOUT_S} ./simba_rtid_exe $i 20 >> res_simba_rtid.csv || true
done
echo "Done!"


echo "Running VFB on all instances..."
echo "instance,formula,time" > res_vfb.csv
for i in ./instances/*.json
do
   timeout ${TIMEOUT_S} ./vfb_exe $i 20 >> res_vfb.csv || true
done
echo "Done!"