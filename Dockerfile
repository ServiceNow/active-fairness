FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.4

# We need to remove PyYAML dist-util.
RUN conda remove PyYAML -y
# We need cmake for SHAP.
RUN pip install cmake "poetry==$POETRY_VERSION"

# Install dependencies.
COPY poetry.lock pyproject.toml /app/
WORKDIR /app
RUN poetry config virtualenvs.create false && \
 poetry install --no-interaction --no-ansi --no-root

# Install the project.
COPY . /app/
RUN poetry install --no-interaction --no-ansi
