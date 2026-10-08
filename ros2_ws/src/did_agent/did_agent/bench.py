"""Run the autonomous agent through scenarios in the kinematic simulator.

    python3 -m did_agent.bench easy medium hard hard@7 medium@3

Names with @ are generated on the fly (difficulty@seed); a range works too
(hard@1-16). With --summary only one aggregate line per difficulty is printed:

    python3 -m did_agent.bench easy@1-8 medium@1-8 hard@1-16 --summary
"""

from __future__ import annotations

import json
import sys
import time


from did_agent.autonomous import AutonomousAgent
from did_agent.scenario_generator import load_named
from did_agent.sim_robot import SimRobot


def run_scenario(name: str, verbose: bool = False, learn: bool = True, **robot_args) -> dict:
    """Run one episode and return a summary."""
    robot = SimRobot(load_named(name), learn=learn, **robot_args)
    agent = AutonomousAgent(robot, log=print if verbose else (lambda message: None))
    started = time.time()
    summary = agent.run()
    judge = robot.judge
    summary.update({
        'scenario': name,
        'sim_seconds': round(robot.sim.t, 1),
        'wall_seconds': round(time.time() - started, 1),
        'distance_m': round(robot.sim.distance, 2),
        'collisions': judge.collisions,
        'false_collects': judge.false_collects,
        'hazard_hits': judge.hazard_hits,
        'score': round(judge.score, 1),
        'finished': judge.finished,
        'learned': learn,
        'hypotheses': robot.adaptation.hypotheses,
    })
    return summary


def expand(name: str) -> list[str]:
    """Expand 'hard@1-16' into ['hard@1', ..., 'hard@16']; other names stay as they are."""
    base, _, seeds = name.partition('@')
    first, dash, last = seeds.partition('-')
    if dash and first.isdigit() and last.isdigit():
        return [f'{base}@{seed}' for seed in range(int(first), int(last) + 1)]
    return [name]


def summarize(label: str, rows: list[dict]) -> dict:
    """One aggregate line for a group of episodes."""
    return {
        'group': label,
        'episodes': len(rows),
        'all_returned': all(r['returned_to_base'] for r in rows),
        'full_collections': sum(r['collected'] == r['samples_total'] for r in rows),
        'collected': sum(r['collected'] for r in rows),
        'samples': sum(r['samples_total'] for r in rows),
        'min_battery': min(r['battery'] for r in rows),
        'hazard_hits': sum(r['hazard_hits'] for r in rows),
        'false_collects': sum(r['false_collects'] for r in rows),
        'collisions': sum(r['collisions'] for r in rows),
        'mean_score': round(sum(r['score'] for r in rows) / len(rows), 1),
    }


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith('-')] or ['easy', 'medium', 'hard']
    learn = '--no-learn' not in sys.argv
    for name in names:
        episodes = expand(name)
        rows = [run_scenario(n, verbose='-v' in sys.argv, learn=learn) for n in episodes]
        if '--summary' in sys.argv:
            print(json.dumps(summarize(name, rows), sort_keys=True))
        else:
            for row in rows:
                print(json.dumps(row, sort_keys=True))


if __name__ == '__main__':
    main()
