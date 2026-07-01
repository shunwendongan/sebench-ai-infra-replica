FROM python:3.11-slim

WORKDIR /app
RUN useradd -m runner
USER runner

CMD ["python", "-c", "print('reproducible work-container answer')"]
