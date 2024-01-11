FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# TODO: move this local

ARG workdir=/var/app
RUN mkdir -p ${workdir}

WORKDIR ${workdir}
COPY . /var/app

# Install Qwen dependencies
RUN python -m pip install -r ./docker/qwen_vl/infer/requirements.txt


EXPOSE 8080
CMD ["uvicorn", "frequency.server.main:app"]
