FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN apt-get update
RUN pip install easydict
RUN pip install tqdm
RUN pip install tensorboardx
RUN pip install thop
RUN pip install torchcontrib
