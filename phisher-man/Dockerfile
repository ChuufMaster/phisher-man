# FROM python:3.12-slim-bookworm
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#
# WORKDIR /app
#
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt
# # RUN pip install --no-cache-dir -r requirements.txt
#
# ADD . /app
#
# WORKDIR /app
# RUN uv sync --locked
#
# COPY . .
#
# ENV PATH="/app/.venv/bin:$PATH"
# # CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.

WORKDIR /app
COPY . .

# Install the application dependencies.
RUN uv sync
# RUN uv pip install fastapi && uv pip install -e .

# Run the application.
# CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]
CMD ["/app/start.sh"]
# CMD [ "sleep", "10000000" ]
