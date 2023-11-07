FROM nvcr.io/nvidia/pytorch:22.12-py3

# ARG USER_NAME=splee
# ARG USER_ID=1007
# ARG GROUP_ID=1007

# RUN groupadd ${USER_NAME} --gid ${GROUP_ID}\ && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash

# USER ${USER_NAME}

RUN apt-get update
RUN pip install easydict
RUN pip install tqdm
RUN pip install tensorboardx
RUN pip install thop
RUN pip install torchcontrib

