ARG PYTHON_VERSION=3.12
FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /uvx /usr/local/bin/

# PROJECT: extra system packages
# RUN apt-get update && apt-get install -y --no-install-recommends <packages> \
#     && rm -rf /var/lib/apt/lists/*

# Mount targets owned by vscode, so named volumes inherit the ownership
RUN install -d -o vscode -g vscode -m 700 /home/vscode/.ssh \
    && install -d -o vscode -g vscode /home/vscode/.cache

# No sudo for the container user
RUN rm -f /etc/sudoers.d/*

USER vscode
