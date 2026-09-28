"""Standalone, single-purpose LLM tasks (table categorization, FAQ generation,
etc.) that don't belong to a specific pipeline like schema_engine or
agent_runtime — each gets its own module here, with prompts factored out
into JSON files under ``ai_agent/prompts/`` (one file per module, same name).
"""
