#!/bin/bash

version=$1
date=$2

# sed -i "s/__version__ = '.*'/__version__ = '$version'/" ./pyproject.toml
sed -i -e "s/version = \".*\"/version = \"$version\"/" ./pyproject.toml

BASEDIR=$(dirname "$0")
cd "$BASEDIR"
cd src/gputls
echo "GTLS_VERSION = '$version'
GTLS_DATE = '$date'" > version.py