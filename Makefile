.PHONY: build up down logs shell test load-test analyze-logs compare-logs

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell-std:
	docker-compose exec std-detection /bin/bash

shell-str:
	docker-compose exec str-recognition /bin/bash

shell-pipeline:
	docker-compose exec pipeline /bin/bash

# 성능 모니터링 및 테스트 관련 명령
load-test:
	docker-compose exec pipeline python -m monitoring.load_test $(ARGS)

analyze-logs:
	docker-compose exec pipeline python -m monitoring.analyze_performance $(ARGS)

compare-logs:
	docker-compose exec pipeline python -m monitoring.analyze_performance --compare $(ARGS)

# 사용 예시:
# make load-test ARGS="--image_dir /path/to/test/images --concurrency 5 --output_dir results"
# make analyze-logs ARGS="--log_files logs/pipeline_performance_*.log --output_dir analysis_results"
# make compare-logs ARGS="--log_files logs/before_mq.log logs/after_mq.log --labels 'Before MQ' 'After MQ' --output_dir comparison_results" 