# run all files called 0xx_*.py in this directory and stop if error
for f in $(ls $(dirname "$0")/0??_*.py | sort); do
    echo "Running $f"
    export MPLBACKEND=Agg
    python $f
    if [ $? -ne 0 ]; then
        echo "Error in $f"
        exit 1
    fi
done
echo "All examples ran successfully."