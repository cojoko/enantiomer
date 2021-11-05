FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "Enantiomer_Chemical_Search", "/bin/bash", "-c"]

COPY . /app

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "Enantiomer_Chemical_Search", "python", "enantiomer.py"]