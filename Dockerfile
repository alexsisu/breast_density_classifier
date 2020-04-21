FROM continuumio/anaconda3 as build-stage


RUN  conda install -y  -c pytorch pytorch
RUN  pip install torchvision
RUN  pip install -U scikit-learn
RUN  pip install imageio
RUN  conda install -y  jupyter

COPY *.* /opt/app/bdc/

COPY run.sh /opt/app/bdc/

ADD saved_models /opt/app/bdc/saved_models

ADD images /opt/app/bdc/images

RUN apt-get update && apt-get install -y procps && apt-get install -y net-tools && apt-get install -y iputils-ping && apt-get install -y less && apt-get install -y vim

WORKDIR /opt/app/bdc

#CMD ["nohup", "/bin/bash", "run.sh","&"]

