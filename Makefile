.PHONY: baseline-1k baseline-3k baseline-6k baseline-12k cva-1k cva-3k cva-6k cva-12k matrix clean-baseline clean-cva clean

baseline-1k:
	./scripts/fortio_job_to_csv.sh baseline-1k 1000 60 60s
baseline-3k:
	./scripts/fortio_job_to_csv.sh baseline-3k 3000 60 60s
baseline-6k:
	./scripts/fortio_job_to_csv.sh baseline-6k 6000 60 60s
baseline-12k:
	./scripts/fortio_job_to_csv.sh baseline-12k 12000 60 60s

cva-1k:
	./scripts/fortio_job_to_csv.sh cva-enabled-1k 1000 60 60s
cva-3k:
	./scripts/fortio_job_to_csv.sh cva-enabled-3k 3000 60 60s
cva-6k:
	./scripts/fortio_job_to_csv.sh cva-enabled-6k 6000 60 60s
cva-12k:
	./scripts/fortio_job_to_csv.sh cva-enabled-12k 12000 60 60s

matrix: baseline-1k baseline-3k baseline-6k baseline-12k cva-1k cva-3k cva-6k cva-12k

clean-baseline:
	kubectl get jobs -o name | grep '^job\.batch/baseline-json' | xargs -r -n1 kubectl delete --ignore-not-found

clean-cva:
	kubectl get jobs -o name | grep '^job\.batch/cva-json' | xargs -r -n1 kubectl delete --ignore-not-found

clean:
	kubectl get jobs -o name | egrep '^job\.batch/(cva-json|baseline-json)' | xargs -r -n1 kubectl delete --ignore-not-found
