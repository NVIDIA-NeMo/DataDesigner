FROM node:26.13.1-bookworm-slim AS pi
RUN npm install --prefix /opt/pi --omit=dev --no-audit --no-fund @earendil-works/pi-coding-agent@1.5.2

FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pi /usr/local/bin/node /opt/pi-runtime/node/bin/node
COPY --from=pi /opt/pi /opt/pi-runtime/pi
WORKDIR /workspace
