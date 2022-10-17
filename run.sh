#!/bin/bash
for dir in examples/*;
do (
  python3 dfa.py "$dir/definition.json" "$dir/strings.json" > "$dir/output.md"
);
done
