mkdir htmls; mv * htmls; mv htmls/output .; cd htmls; for i in *; do cp ../output/$i.json $i; cd $i; python ../../../render_json.py $i.json; cd ../; done
