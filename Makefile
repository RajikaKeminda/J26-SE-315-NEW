# Common tasks.  `make help` lists them.
PY ?= python3
KIND_CLUSTER ?= autoscaler

.PHONY: help install db-up test train ablate pareto report reproduce calibrate live manifests dashboard docker-build kind-up kind-load deploy e2e kind-down compose-up compose-down

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## install autoscaler + test dependencies
	$(PY) -m pip install -r autoscaler/requirements-dev.txt

db-up:           ## start the PostgreSQL that stores the agent's history
	bash scripts/22_db_up.sh

test:            ## unit + integration tests (no cluster needed; the database tests need TEST_DATABASE_URL)
	cd autoscaler && $(PY) -m pytest -q

train:           ## train the MO-TD3 agent and every ablation arm on the calibrated simulator
	bash scripts/06_train_all_arms.sh

ablate:          ## evaluate every controller on identical traffic (6-service + unseen 14-service app)
	bash scripts/09_run_ablation.sh

pareto:          ## preference sweep of ONE MO-TD3 network vs the HPA tuning sweep -> Pareto fronts
	bash scripts/12_run_pareto.sh

report:          ## regenerate the figures and the README result tables from results/*.json
	cd autoscaler && $(PY) -m src.evaluation.plots && $(PY) -m src.evaluation.report

reproduce:       ## re-create every simulator result from scratch (train, evaluate, Pareto, figures, README)
	bash scripts/17_reproduce_results.sh

calibrate:       ## re-measure the real service on your cluster and re-fit the simulator
	bash scripts/13_calibrate.sh

live:            ## real-cluster head-to-head: static / tuned HPA / MO-TD3 with in-cluster load
	bash scripts/15_live_experiment.sh

manifests:       ## regenerate k8s/01-services from autoscaler/src/topologies.py
	$(PY) scripts/gen_manifests.py

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
