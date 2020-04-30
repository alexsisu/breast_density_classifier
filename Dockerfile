FROM continuumio/anaconda3 as build-stage


RUN  conda install -y  -c pytorch pytorch
RUN  pip install torchvision
RUN  pip install -U scikit-learn
RUN  pip install imageio
RUN  conda install -y  jupyter

COPY *.* /opt/app/bdc/

ADD saved_models /opt/app/bdc/saved_models

ADD images /opt/app/bdc/images

RUN apt-get update && apt-get install -y procps && apt-get install -y net-tools && apt-get install -y iputils-ping && apt-get install -y less && apt-get install -y vim

WORKDIR /opt/app/bdc

ENV TINI_VERSION v0.6.0

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini

RUN chmod +x /usr/bin/tini

EXPOSE 15432

RUN python simple_server.py
#ENTRYPOINT ["/usr/bin/tini", "--"]

#EXPOSE 8888

#CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--no-browser", "--ip=0.0.0.0","--NotebookApp.token=''"]

