#!/bin/sh
# Regenerate the model, checks, stencils and report. Edit helmet/params.py first.
cd "$(dirname "$0")" && exec blender --background --factory-startup --python-exit-code 1 --python build.py
