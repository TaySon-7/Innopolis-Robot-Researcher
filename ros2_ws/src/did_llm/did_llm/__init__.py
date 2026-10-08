"""LLM layer of the DID Hack agent.

The package is split into a ROS-free core and a thin ROS wrapper:

* :mod:`did_llm.schemas`      - subgoal vocabulary and plan validation
* :mod:`did_llm.prompts`      - planner and hypothesis prompts
* :mod:`did_llm.llm_client`   - OpenAI-compatible client with a call budget
* :mod:`did_llm.rule_planner` - deterministic reference planner and fallback
* :mod:`did_llm.planner_model` - orchestration, ROS-independent and testable
* :mod:`did_llm.journal`      - exchange log and hypothesis ledger
* :mod:`did_llm.planner_node` - ROS node wrapper around the core
"""
