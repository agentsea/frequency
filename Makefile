
.PHONY: gen-api
gen-api:
	fastapi-codegen --input ./frequency/api/v1/server.yaml --output ./frequency/api/v1/server/ --generate-routers

.PHONY: gen-client
gen-client:
	autorest --input-file=./frequency/api/v1/server.yaml --python --output-folder=./frequency/client/v1

.PHONY: serve
serve:
	poetry run uvicorn frequency.server.main:app --reload
