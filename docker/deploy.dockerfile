FROM spectreteam/python_msi:v5.0.0

ARG USERNAME=msi
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME &&\
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME &&\
    apt-get update &&\
    apt-get install -y procps sudo &&\
    rm -rf /var/lib/apt/lists/* &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
    rm /requirements.txt

RUN mkdir /app

WORKDIR /app

COPY bin bin

COPY components components

COPY pipeline pipeline

COPY plot.py plot.py

COPY test test

RUN python -m unittest discover

RUN rm -rf test

COPY luigi.cfg luigi.cfg

# Luigi scheduler port
EXPOSE 8082

# Data mount point
VOLUME /data

# History preservation
VOLUME /luigi

COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

USER $USERNAME
