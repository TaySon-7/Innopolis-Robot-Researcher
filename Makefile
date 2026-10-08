.PHONY: build up down restart logs shell teleop topics pose scan battery sensor score events goto collect finish auto stop demo dashboard scenario scenario-check plan bench check test test-fast

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

events:
	docker compose exec sim /did-entrypoint.sh ros2 topic echo /did/events --field data

# Usage: make goto X=0.55 Y=0.55
goto:
	docker compose exec sim /did-entrypoint.sh ros2 run did_agent goto --x $(X) --y $(Y)

collect:
	docker compose exec sim /did-entrypoint.sh ros2 service call /did/collect std_srvs/srv/Trigger

finish:
	docker compose exec sim /did-entrypoint.sh ros2 service call /did/finish std_srvs/srv/Trigger

# Autonomous agent without an LLM: one whole episode, prints a summary at the end.
auto:
	docker compose exec sim /did-entrypoint.sh ros2 run did_agent command auto --wait

stop:
	docker compose exec sim /did-entrypoint.sh ros2 run did_agent command stop

# Start everything and open the dashboard: make demo SCENARIO=hard
demo: up dashboard

dashboard:
	@echo "Dashboard: http://localhost:8080"
	-@open http://localhost:8080 2>/dev/null || xdg-open http://localhost:8080 2>/dev/null || true

# Generate a scenario file: make scenario DIFF=hard SEED=7  ->  scenarios_custom/hard-7.yaml
# Run it:   SCENARIO_FILE=/scenarios_custom/hard-7.yaml make demo      (or just: SCENARIO=hard@7 make demo)
scenario:
	docker compose run --rm --no-deps sim ros2 run did_agent generate_scenario \
		--difficulty $(or $(DIFF),medium) --seed $(or $(SEED),1) --out /scenarios_custom/$(or $(DIFF),medium)-$(or $(SEED),1).yaml

# make scenario-check FILE=/scenarios_custom/mine.yaml
scenario-check:
	docker compose run --rm --no-deps sim ros2 run did_agent generate_scenario --check $(FILE)

# Usage: make plan P='{"plan_id":"p1","subgoals":[{"type":"goto","x":0.5,"y":0.5}]}'
plan:
	docker compose exec sim /did-entrypoint.sh ros2 run did_agent send_plan '$(P)' --wait

# Whole episodes in the kinematic simulator (seconds, no Gazebo): make bench S="easy hard"
bench:
	docker compose run --rm --no-deps sim bash -lc "python3 -m did_agent.bench $(S)"

check:
	docker compose exec sim /did-entrypoint.sh ros2 run did_judge level0_check

# Unit tests inside the image (no simulator needed).
test:
	docker compose run --rm --no-deps sim bash -lc \
		"cd /opt/did_ws/src && python3 -m pytest -q did_judge/test did_agent/test"

# The same tests on the host, if it has python3, numpy, pyyaml and pytest.
test-fast:
	cd ros2_ws/src && python3 -m pytest -q did_judge/test did_agent/test
