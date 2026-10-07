.PHONY: build up down restart logs shell teleop topics pose scan battery sensor score check test

build:
	docker compose build

up:
	docker compose up --detach --wait

down:
	docker compose down

restart: down up

logs:
	docker compose logs --follow sim

shell:
	docker compose exec sim /did-entrypoint.sh bash

teleop:
	docker compose exec sim /did-entrypoint.sh ros2 run turtlebot3_teleop teleop_keyboard

topics:
	docker compose exec sim /did-entrypoint.sh ros2 topic list

pose:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /odom --once

scan:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /scan --once

battery:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /did/battery --once

sensor:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /did/sample_sensor --once

score:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /did/score --once

check:
	docker compose exec sim /did-entrypoint.sh ros2 run did_judge level0_check

test:
	docker compose run --rm --no-deps sim bash -lc \
		"cd /opt/did_ws && python3 -m pytest -q src/did_judge/test"
