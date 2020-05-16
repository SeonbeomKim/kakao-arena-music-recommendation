FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

ENV PYTHONPATH "${PYTHONPATH}:/home/edwin/workspace/kakao-arena-music-recommendation:/home/edwin/workspace"
ENV TRAINER_DIR "/home/edwin/workspace/kakao-arena-music-recommendation"
WORKDIR ${TRAINER_DIR}

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
CMD python trainer/wmf_trainer.py