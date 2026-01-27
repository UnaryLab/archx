if [ -z "$1" ]; then
    echo "Usage: $0 <cs>"
    exit 1
fi

cs="$1"

# Create runtime directory if it doesn't exist
mkdir -p agraph_casestudy/runtime

# Record start time with nanosecond precision
start_time=$(date +%s.%N)

# Run the three commands
python -m "agraph_casestudy.${cs}.description"
# bash src/archx/run_archx.sh "agraph_casestudy/${cs}/description/runs.txt"
# python -m "agraph_casestudy.${cs}.query.query"

# # Record end time with nanosecond precision
# end_time=$(date +%s.%N)

# # Calculate total runtime
# runtime=$(echo "$end_time - $start_time" | bc)

# # Save runtime to file
# echo "Total runtime: ${runtime} seconds" > "agraph_casestudy/runtime/${cs}.txt"