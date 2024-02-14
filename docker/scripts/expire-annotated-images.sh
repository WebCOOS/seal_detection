#!/bin/bash

if [ -z "${OUTPUT_DIRECTORY}" ]; then
    echo "Must set OUTPUT_DIRECTORY environment variable" 1>&2;
    exit 1;
fi

if [ ! -d "${OUTPUT_DIRECTORY}" ]; then
    echo "OUTPUT_DIRECTORY ('${OUTPUT_DIRECTORY}') must exist and be a directory." 1>&2;
    exit 2;
fi

# Minutes after which an output annotation image will expire
# 1440 = 24 hours
EXPIRE_MINUTES=1440

while read -r line; do
    if [ -n "$line" -a -e "$line" ]; then
        echo "expiring the file '$line'";
        rm -v "$line";
    else
        echo "No matching files, wait...";
    fi
done < <(
    find "${OUTPUT_DIRECTORY}" -type f -and -mmin "+${EXPIRE_MINUTES}" -and -name "*.jpg"
);
