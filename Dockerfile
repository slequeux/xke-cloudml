FROM continuumio/anaconda

MAINTAINER Sylvain Lequeux<slequeux@gmail.com>

# Install GCloud
WORKDIR /opt/gcloud
RUN /bin/bash -c 'wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-200.0.0-linux-x86_64.tar.gz'
RUN /bin/bash -c 'tar xzvf google-cloud-sdk-200.0.0-linux-x86_64.tar.gz'
RUN /bin/bash -c '/opt/gcloud/google-cloud-sdk/install.sh -q'
RUN /bin/bash -c 'echo "source /opt/gcloud/google-cloud-sdk/completion.bash.inc" >> ~/.bashrc'
RUN /bin/bash -c 'echo "source /opt/gcloud/google-cloud-sdk/path.bash.inc" >> ~/.bashrc'
RUN /bin/bash -c '/opt/gcloud/google-cloud-sdk/bin/gcloud components update'

# Configure Conda
RUN /bin/bash -c 'conda update -n base conda'
RUN /bin/bash -c 'conda create -n xke-cloudml -q python=2.7'
RUN /bin/bash -c 'echo "source activate xke-cloudml" >> ~/.bashrc'

# Install python dependencies
COPY ./requirements.txt /tmp
RUN /bin/bash -c 'source ~/.bashrc && pip install --upgrade pip && pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt'

VOLUME ["/opt/demo"]
WORKDIR /opt/demo

RUN /bin/bash -c 'echo alias glogin=\"gcloud auth login --no-launch-browser\" >> ~/.bashrc'
RUN /bin/bash -c 'echo alias gcreds=\"gcloud auth application-default login --no-launch-browser\" >> ~/.bashrc'

ENTRYPOINT /bin/bash