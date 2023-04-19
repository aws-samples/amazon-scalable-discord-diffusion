FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y wget git && apt-get clean

RUN git clone https://github.com/db0/nataili.git .
# Check out a specific version of the above repository
RUN git checkout 6c2f1862bacf25b6bc74e95e3174ca45a116f85b 
RUN echo "boto3>=1.21.32">>requirements.txt

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda

# Add miniconda to the PATH
ENV PATH=/miniconda/bin:$PATH

# Update conda and install any necessary packages
RUN conda update --name base --channel defaults conda && \
    conda env create -f /app/environment.yaml --force && \
    conda clean -a -y

# Install conda environment into container so we do not need to install every time.
ENV ENV_NAME ldm

COPY ecs_run.py /app/

SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ldm", "python", "ecs_run.py"]
