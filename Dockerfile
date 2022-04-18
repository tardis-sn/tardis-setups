FROM continuumio/miniconda3:4.10.3p0-alpine

WORKDIR /app

# Copy the conda environment file
COPY tardis_env3.yml .

# Installing git and openssh
RUN apk add --no-cache git
RUN apk add --no-cache openssh

# Update conda
RUN conda update conda

## Create an environment
RUN conda env create -f tardis_env3.yml

## Activate the environment
SHELL ["conda", "run", "-n", "tardis", "/bin/bash", "-c"]


## Clone tardis repository
RUN git clone https://github.com/tardis-sn/tardis.git /app/tardis

## Change the working directory to the repository
WORKDIR /app/tardis

# Install TARDIS
RUN python setup.py install
