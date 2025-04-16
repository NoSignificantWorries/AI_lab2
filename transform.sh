#!/bin/bash

find resources/water_v2/Annotations -name "*.jpg" -print0 | while IFS= read -r -d $'\0' file; do
  convert "$file" "${file%.jpg}.png"
  rm "$file"
done
