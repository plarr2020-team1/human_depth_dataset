#!/bin/bash

files=(2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0013
2011_09_26_drive_0020
2011_09_26_drive_0023
2011_09_26_drive_0036
2011_09_26_drive_0079
2011_09_26_drive_0095
2011_09_26_drive_0113
2011_09_28_drive_0037
2011_09_29_drive_0026
2011_09_30_drive_0016
2011_10_03_drive_0047)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done
