FROM python:3.8

WORKDIR /app
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"
COPY . /app

RUN pip install -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]

