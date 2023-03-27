FROM python:3.9 
#line sets the base image to use for the Docker image. In this case, it is the official Python 3.9 image

COPY . /searchengine
#copies all the files in the current directory (represented by .) to a new directory /searchengine in the Docker image.

WORKDIR /searchengine
#line sets the working directory for the Docker image to /searchengine.

RUN pip3 install --upgrade pip
# line upgrades pip, the package installer for Python.

RUN pip3 install -r requirements.txt
# line installs the Python packages listed in the requirements.txt file, which should be present in the same directory as the Dockerfile.
EXPOSE 8080
# line exposes port 8080 to the host system.
CMD ["python","app.py"]
#line specifies the command to run when the Docker container is started. In this case, it runs the Python
# script app.py.



