DOCKER_IMG=fmrihandbook
all: Introduction ImageProcessing

Introduction:
	docker run -it -v ${PWD}:/handbook ${DOCKER_IMG} python notebooks/Introduction/Introduction.py


docker-shell:
	docker run -it -v ${PWD}:/handbook --entrypoint=bash ${DOCKER_IMG}

