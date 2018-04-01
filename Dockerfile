FROM python:slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update -qq && \
	DEBIAN_FRONTEND=noninteractive apt-get install -qq \
		python3-tk \
		xvfb \
		curl && \
	curl -OJL https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 && \
	pip install --no-cache-dir -r requirements.txt && \
	apt-get remove --purge -qq curl && \
	apt-get autoremove --purge -qq && \
	apt-get clean -qq && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /init /root/.cache

COPY . .

WORKDIR /data

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["-n"]
