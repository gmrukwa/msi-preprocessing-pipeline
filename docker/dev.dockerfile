FROM spectreteam/python_msi:v5.0.0

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
    rm /requirements.txt

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for non-root user
    && apt-get update \
    && apt-get install -y \
        sudo \
        iproute2 \
        procps \
        gcc \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install memory_profiler

# Luigi scheduler port
EXPOSE 8082

# Data mount point
VOLUME /data

RUN mkdir /luigi
