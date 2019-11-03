#!/bin/bash
sudo luigid --background
luigi --module pipeline PreprocessingPipeline --datasets "$@"
