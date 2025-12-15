SECONDS=0
bash zoo/llm/llm_script.sh
elapsed=$SECONDS
elapsed=$((end - start))

printf "\nTotal runtime: %02d:%02d:%02d\n" \
  $((elapsed/3600)) \
  $(((elapsed%3600)/60)) \
  $((elapsed%60))