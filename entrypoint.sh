#!/bin/bash
luigid --background
luigi --module pipeline PreprocessingPipeline --datasets "$@"
