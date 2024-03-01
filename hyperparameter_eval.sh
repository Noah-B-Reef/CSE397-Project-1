#!/bin/bash

# Define the range of values for i
start=1
end=7  # Change this to the desired end value

# Loop over the range of values for i
for ((i=$start; i<=$end; i++)); do
    echo "Executing python -u eval_model.py $i"
    python -u eval_model.py models/HyperparameterSearch/IsothermalAtmoAlt_6x128_100x50_seed\=$i data/HPSearch/eval_data/eval_IsothermalAtmoAlt_6x128_100x50_seed\=$i.h5
done
