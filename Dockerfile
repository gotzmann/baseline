# Ubuntu 20.04.6 LTS
# Python 3.9.16

# -- Build, tag and push image
# sudo docker build --tag supermachina:0.2 .
# sudo docker tag supermachina:0.1 cr.msk.sbercloud.ru/aijcontest/supermachina:0.2
# sudo docker push cr.msk.sbercloud.ru/aijcontest/supermachina:0.2

# -- Build for multi platforms
# sudo docker buildx build --platform linux/amd64 -f ./Dockerfile --tag supermachina:0.2 .

# -- Show and prune Docker cache
# sudo docker system df
# sudo docker builder prune

# -- Show and remove unused images
# sudo docker image ls
# sudo docker image rm supermachina:0.1

# -- Show TOP 20 biggest files and folders
# sudo du -ah / | sort -rh | head -n 20

# -- Reset GPU
# nvidia-smi --gpu-reset

# -- Show and kill processes using GPU
# lsof | grep /dev/nvidia

# FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

# MLSPACE_IMAGE_PARENT=nvidia/cuda:-devel-ubuntu20.04
# MLSPACE_IMAGE_NAME=cuda11.7-torch2
FROM cr.msk.sbercloud.ru/aijcontest_official/fbc3_0:0.1 as base
USER root
WORKDIR /app

COPY model.gguf /app/model.gguf
COPY imagebind_huge.pth /app/imagebind_huge.pth
COPY projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind

# RUN apt update -y && \
#     apt upgrade -y && \
#     apt install -y mc nano git htop lsof make build-essential

# RUN wget https://golang.org/dl/go1.20.linux-amd64.tar.gz && \
#     tar -xf go1.20.linux-amd64.tar.gz -C /usr/local

# RUN git clone https://github.com/gotzmann/llamazoo.git && \
#     cd ./llamazoo && \
#     LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j cuda && \
#     mkdir /app && \
#     cp llamazoo /app/llamazoo && \
#     chmod +x /app/llamazoo

# json, time, traceback : standard python lib
# numpy : Requirement already satisfied: numpy in /home/user/conda/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.24.1)



# RUN pip install https://github.com/enthought/mayavi/zipball/master
# RUN pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo
# RUN pip install --no-cache-dir -r requirements.txt

# -- See standard Python libs: https://docs.python.org/3/library/index.html
RUN apt update && apt install -y --no-install-recommends python3-pip
# RUN pip install requests
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

FROM base
USER root
WORKDIR /app

#COPY ./Llama-2-7B-fp16 ./Llama-2-7B-fp16

# COPY --from=base /app/model.gguf /app/model.gguf
# COPY --from=base /app/imagebind_huge.pth /app/imagebind_huge.pth
# COPY --from=base /app/projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind

COPY config.yaml        /app/config.yaml
COPY llamazoo           /app/llamazoo
# RUN chmod +x            /app/llamazoo

# DEBUG
# ENTRYPOINT [ "./llamazoo", "--server", "--debug" ]

# USER jovyan
# WORKDIR /home/jovyan
