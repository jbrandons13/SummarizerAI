#!/bin/bash
echo "Waiting for runner to finish..."
while pgrep -f "pipeline/facet/runner.py" > /dev/null; do
    sleep 5
done

echo "Runner finished. Running verify and scorer..."
PYTHONPATH=. python pipeline/facet/verify_g0.py > verify_out.txt
PYTHONPATH=. python pipeline/facet/scorer.py > scorer_out.txt

echo -e "\n\n## G0 Addendum: Baseline Reproduction (Geology Sweep)\n" >> runs/G0_REPORT.md

cat verify_out.txt >> runs/G0_REPORT.md
echo -e "\n" >> runs/G0_REPORT.md
cat scorer_out.txt >> runs/G0_REPORT.md

echo "G0 Addendum appended to runs/G0_REPORT.md"
