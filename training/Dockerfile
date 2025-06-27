#  Use Python 3.12 as base image
FROM python:3.12

#  Create necessary directories inside container
RUN mkdir -p /app/src/
RUN mkdir -p /app/data/
RUN mkdir -p /app/model/

#  Copy code and dependency files
COPY requirements.txt /app/src/
COPY main.py /app/src/
COPY irisdataset.csv /app/data/   

#  Install Python dependencies
RUN pip3 install -r /app/src/requirements.txt

# Give read/write/execute permissions
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app

# Run your script
CMD ["python", "/app/src/main.py"]
