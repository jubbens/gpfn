FROM continuumio/miniconda3

# Copy only the git repo files (including environment.yml) to the image
COPY .git app/.git
COPY .gitignore app/.gitignore
RUN git --git-dir=/app/.git --work-tree=/app ls-files -z | xargs -0 git --git-dir=/app/.git --work-tree=/app checkout HEAD --

# Copy the models over
COPY deploy /deploy

# Install conda environment
RUN conda env create -f app/environment-mini.yml && conda clean -afy

ENV PYTHONPATH=/app

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "gpfn", "python", "/app/docker/entry.py"]
