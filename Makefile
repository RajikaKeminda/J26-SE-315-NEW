# Common tasks.  `make help` lists them.
PY ?= python3
KIND_CLUSTER ?= autoscaler

.PHONY: help install test train ablate pareto dashboard docker-build kind-up kind-load deploy e2e kind-down compose-up compose-down

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## install autoscaler + test dependencies
	$(PY) -m pip install -r autoscaler/requirements-dev.txt

test:            ## unit + integration tests (no cluster needed)
	cd autoscaler && $(PY) -m pytest -q

train:           ## train all four ablation arms on the simulator
	bash scripts/06_train_all_arms.sh

ablate:          ## evaluate every arm + HPA/static on identical traffic
	bash scripts/09_run_ablation.sh

pareto:          ## reward-weight sweep -> cost/latency Pareto front
	bash scripts/12_run_pareto.sh

dashboard:       ## demo-mode dashboard on :8080 (no cluster)
	bash scripts/10_run_dashboard.sh

docker-build:    ## build the service and autoscaler images
	docker build -t k8s-autoscaler/generic-service:latest services/generic-service
	docker build -t k8s-autoscaler/autoscaler:latest autoscaler

kind-up:         ## create a local kind cluster
	bash scripts/00_kind_up.sh $(KIND_CLUSTER)

kind-load:       ## load both images into the kind cluster
	kind load docker-image k8s-autoscaler/generic-service:latest k8s-autoscaler/autoscaler:latest --name $(KIND_CLUSTER)

deploy:          ## deploy observability, services and the in-cluster autoscaler
	bash scripts/03_deploy_all.sh
	bash scripts/04_verify_deployment.sh

e2e:             ## smoke test against the running cluster (traffic -> metrics -> scaling)
	bash scripts/14_e2e_smoke.sh

kind-down:       ## delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

compose-up:      ## whole stack without Kubernetes
	docker compose up --build -d

compose-down:
	docker compose down
