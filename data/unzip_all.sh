#!/bin/bash

for file in *.zip; do
  case "$file" in
    *.zip)
      unzip "$file" -d "${file%.zip}"
      ;;
    *.tar.gz)
      mkdir -p "${file%.tar.gz}" && tar -xzf "$file" -C "${file%.tar.gz}"
      ;;
    *.tar.bz2)
      mkdir -p "${file%.tar.bz2}" && tar -xjf "$file" -C "${file%.tar.bz2}"
      ;;
    *.gz)
      gunzip "$file"
      ;;
  esac
done