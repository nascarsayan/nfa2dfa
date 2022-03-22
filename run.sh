#!/bin/bash
for dir in input/*;
do (
  python dfa.py "$dir/definition.json" "$dir/strings.json" > "$dir/output.md"
);
done
