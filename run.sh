#!/bin/bash
for dir in input/*;
do (
  python3 dfa.py "$dir/definition.json" "$dir/strings.json" > "$dir/output.md"
);
done
