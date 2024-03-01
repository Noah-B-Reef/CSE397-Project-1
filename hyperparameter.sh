#!/bin/bash

# Define the range of values for i
start=1
end=7  # Change this to the desired end value

# Loop over the range of values for i
for ((i=$start; i<=$end; i++)); do
    echo "Executing python -u PINNS2.py $i"
    python -u PINNS2.py $i
done
