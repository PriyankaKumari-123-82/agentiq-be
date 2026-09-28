def format_schema(schema_json: dict) -> str:
    schema_lines = []

    for table_name, table_data in schema_json.items():
        columns = table_data.get("columns", [])
        primary_keys = set(table_data.get("primary_keys", []))

        foreign_keys = {
            fk["column"]: f'{fk["references_table"]}.{fk["references_column"]}'
            for fk in table_data.get("foreign_keys", [])
        }

        formatted_columns = []

        for col in columns:
            col_name = col["column"]
            col_type = col["type"]

            parts = [f"{col_name}:{col_type}"]

            if col_name in primary_keys:
                parts.append("PK")

            if col_name in foreign_keys:
                parts.append(f"FK->{foreign_keys[col_name]}")

            formatted_columns.append(" ".join(parts))

        table_schema = (
            f"{table_name}("
            f"{', '.join(formatted_columns)}"
            f")"
        )

        schema_lines.append(table_schema)

    final_schema = 'schema="""\n'
    final_schema += "\n".join(schema_lines)
    final_schema += '\n"""'

    return final_schema
