FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    patchelf \
    gcc \
    g++ \
    gfortran \
    make \
    ccache \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

RUN python3 -m venv venv
RUN . venv/bin/activate && pip install --no-cache-dir nuitka PyQt6 pyqt6-tools matplotlib scikit-learn nltk pandas qasync gensim

VOLUME ["/app/dist"]
RUN echo '#!/bin/sh\n. venv/bin/activate\nexport PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")\nnuitka --standalone --lto=yes --onefile --follow-imports --enable-plugin=pyqt6 PalindromiPeli.py --output-dir=/app/dist' > /app/build.sh \
    && chmod +x /app/build.sh

CMD ["/app/build.sh"]
