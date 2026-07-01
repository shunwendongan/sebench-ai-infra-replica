FROM python:3.11-slim

WORKDIR /judge
RUN useradd -m judge
USER judge

CMD ["python", "-c", "print('judge container ready')"]
