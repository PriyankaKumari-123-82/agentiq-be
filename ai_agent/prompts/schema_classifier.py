"""Prompts used by ``ai_agent.workflows.schema_classifier.schema_classifier``.

Both are user-role message templates with ``str.format()`` placeholders —
callers must ``.format(...)`` the ``content`` before sending.
"""

SCHEMA_CLASSIFIER_PROMPT = {
    "role": "user",
    "content": (
        "You are a database domain expert. Below is every table in a database, each with "
        "its business summary and keywords. Group these tables into real, business-"
        "meaningful categories (e.g. \"complaints\", \"spare parts\", \"access control\", "
        "\"employees\" etc.) based on what they are actually used for. The \"domain hint\" shown "
        "for each table is a coarse fallback label (only a handful of generic buckets) — "
        "do not just copy it, use it only as a weak signal.\n\n"
        "Rules:\n"
        "- Every table must appear in exactly one category.\n"
        "- Prefer several specific categories over one broad catch-all.\n"
        "- Group tables that are clearly part of the same subsystem even if their coarse "
        "domain hints differ.\n"
        "- Never create a catch-all category (e.g. \"Miscellaneous\", \"Other\", \"General\", "
        "\"Uncategorized\"). Every table must go into a specific, meaningfully-named "
        "category based on what it's actually used for, even if that category ends up "
        "with only one or two tables in it.\n"
        "- Order the \"categories\" array from most to least business-critical (the "
        "category a typical business user would care about most first — e.g. core "
        "transactional/operational data before rarely-used lookup/admin tables).\n\n"
        "Tables:\n"
        "{tables}\n\n"
        "Respond ONLY with a JSON object of this exact shape, no markdown fences:\n"
        "{{\"categories\": [\n"
        "  {{\"name\": \"<short category name>\", \"description\": \"<1-sentence description>\", \"tables\": [\"<table_name>\", ...]}}\n"
        "]}}"
    )
}

SCHEMA_CLASSIFIER_QUESTION_PROMPT = {
    "role": "user",
    "content": (
        "You are helping business users explore a database category called \"{name}\" "
        "({description}).\n\n"
        "Here are the tables in this category (columns, types, and possible enum values):\n\n"
        "{ddl}\n\n"
        "Write 4-5 concise, realistic questions a business user could ask about this "
        "data (e.g. \"How many complaints were logged today?\"). Ground each question in "
        "the actual columns/enum values shown above — don't invent fields that aren't "
        "there.\n\n"
        "Respond ONLY with a JSON object of this exact shape, no markdown fences:\n"
        "{{\"questions\": [\"<question 1>\", \"<question 2>\", ...]}}"
    )
}
