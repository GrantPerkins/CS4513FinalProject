FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN pip install tensorflow-datasets matplotlib

ENV PATH="/opt/ml/code:${PATH}"

COPY /mnist /opt/ml/code
WORKDIR /opt/ml/code