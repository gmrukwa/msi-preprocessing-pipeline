#!/bin/bash

# start the scheduler
luigid &>/dev/null &

# check if we should keep scheduler alive after tasks finish
args=("$@")
while [[ ! -z "$2" ]]; do shift; done;
last=$1
if [ "$last" = "--keep-alive" ]; then
  args=("${args[@]::${#args[@]}-1}")
fi

# launch the preprocessing
PYTHONPATH='.' luigi --module pipeline PreprocessingPipeline --datasets ${args[*]}

# prevent closing
if [ "$last" = "--keep-alive" ]; then
  read -p "Press [Enter] key to close..."
fi

# kill the scheduler
pkill luigid
