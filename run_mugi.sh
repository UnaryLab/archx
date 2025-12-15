start=`date +%s`
bash zoo/llm/llm_script.sh
end=`date +%s`
runtime=$((end-start))
echo "Total runtime: $runtime seconds."