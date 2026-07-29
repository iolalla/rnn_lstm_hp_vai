# Load environment variables from .env file if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

IMAGE_URI ?= gcr.io/banca-march/rnn_lstm_vai:hypertune

help: ## show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

build: ## Build the Docker image (must be run from repository root)
	docker build -f Dockerfile -t $(IMAGE_URI) .

push: ## Push the Docker image to GCR
	docker push $(IMAGE_URI)

job: ## Run the orchestrator job script locally to submit to Vertex AI
	python job.py

run-local: ## Run the container locally using host network to connect to BigQuery emulator
	docker run --rm -it --net=host \
		-e ISLOCAL=true \
		-e BIGQUERY_EMULATOR_HOST=localhost:9050 \
		$(IMAGE_URI) \
		--epochs=2 \
		--learning_rate=0.001 \
		--units=32 \
		--activation=relu \
		--dropout_rate=0.1 \
		--activation_output=linear \
		--filedata=https://storage.googleapis.com/ibex35/data/IBEX-1994-2020.csv

run-local-script: ## Run the training script locally using python and uv
	uv run python3 trainer/task.py \
		--epochs=2 \
		--learning_rate=0.001 \
		--units=32 \
		--activation=relu \
		--dropout_rate=0.1 \
		--activation_output=linear \
		--filedata=data/reall-complete-IBEX-2021.csv \
		--ticker=SAN.MC

all: build push ## Build and push the image
