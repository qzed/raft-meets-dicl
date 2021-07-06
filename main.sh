#!/bin/sh

PYTHON="python3"
MAINFILE="src/main.py"

BASEDIR="$(dirname "${0}")"

${PYTHON} "${BASEDIR}/${MAINFILE}" "${@}"
