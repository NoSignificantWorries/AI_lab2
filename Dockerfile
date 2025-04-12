FROM nosignificantworries/torch:base AS base

WORKDIR /home/app

COPY *.py .
