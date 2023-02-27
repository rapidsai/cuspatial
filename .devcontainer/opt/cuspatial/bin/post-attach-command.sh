#! /usr/bin/env bash

# Source this call in case we're running in Codespaces.
#
# Codespaces runs the "postAttachCommand" in an interactive login shell.
# Once "postAttachCommand" is finished, the terminal is relenquished to
# the user. Sourcing here ensures the new conda env is already activated
# in the shell for the user.
source rapids-make-${PYTHON_PACKAGE_MANAGER}-env;
