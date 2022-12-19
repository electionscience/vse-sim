#!/bin/bash
svgs="*.svg"
for svg in $svgs
do
  echo "Converting $svg"
  /Applications/Inkscape.app/Contents/MacOS/inkscape --export-background-opacity=0 --export-type=png   --export-filename="pngs/$svg.png" "$svg"
done
