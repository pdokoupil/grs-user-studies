FROM continuumio/miniconda3:4.12.0

# RUN mkdir -p /app/src
WORKDIR /app
#ENV HOME=/app

# COPY requirements.txt /app/requirements.txt
COPY requirements.txt requirements.txt
# COPY /src /app/src # Bind volume instead (no need to rebuild)
# Once deploying, we should copy instead

RUN conda update conda --yes \ 
    && conda config --add channels conda-forge \
    && conda create --name app_env python=3.9 \
    && conda install --yes --file requirements.txt




# RUN git clone https://github.com//sisinflab/elliot.git

RUN pip install --upgrade pip
# RUN pip install -e ./elliot --verbose
RUN pip install caserecommender
RUN pip install Flask-PluginKit

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]