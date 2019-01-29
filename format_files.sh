#!/bin/bash
folder=.
find "${folder}" -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -not -path "${exclude_folder}" -prune -exec clang-format-6.0 -style=file -i {} \;

