#!/bin/bash

# Install app with icon on Linux

cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m install_desktop ~/opt/ir_tool/ir_tool_UI UI/ir_tool_64.png "IR Tool" "Batch deconvolve / process impulse response tool"