#!/bin/bash
cd svg_output
pwd
svgs=`echo *.svg`
echo $svgs
cd ..
echo "butt..."
echo $svgs
for svg in $svgs
do
  echo "Converting $svg"
  /Applications/Inkscape.app/Contents/MacOS/inkscape --export-background-opacity=0 --export-type=png   --export-filename="pngs/$svg.png" "svg_output/$svg"
done
