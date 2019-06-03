#!/bin/bash
folder=1

for i in $(docker ps -a -q); do
    docker cp  $i:/usr/src/app/output/lipizzaner_gan ./$folder/
    folder=$(( folder + 1 ))

done
