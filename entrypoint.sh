#!/bin/bash
luigid &>/dev/null &
PYTHONPATH='.' luigi --module pipeline PreprocessingPipeline --datasets "$@"
pkill luigid
