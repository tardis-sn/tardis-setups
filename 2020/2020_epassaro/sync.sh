#! /bin/bash

if [ "$1" == "" ]; then
    echo; echo "Usage:  sync.sh <MSU_NETID>"
    exit 1
fi

rsync -ave ssh $1@rsync.hpcc.msu.edu:/mnt/home/$1/Documents/tardis-setups/2020/2020_epassaro/Output .

exit 0
