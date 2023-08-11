#!/bin/bash
docker run --init --rm -ti -p 8888:8888 --name mealor -e JUPYTER_ENABLE_LAB=yes -e CHOWN_HOME=yes -e CHOWN_EXTRAOPTS='-hR' --user root -v "$(pwd)":/home/jovyan/shared mealor:latest

