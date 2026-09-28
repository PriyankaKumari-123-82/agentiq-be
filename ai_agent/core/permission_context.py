"""Permission-aware AI with data isolation for users.

This module provides:
- Row-level security (RLS) context injection
- Tenant isolation for multi-tenant scenarios
- Data filtering based on user permissions
- Query rewriting with security predicates
"""
import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    """Permission levels for data access."""
    NONE = "none"
    OWN = "own"  # Only own data
    TEAM = "team"  # Team/org data
    DEPARTMENT = "department"
    FULL = "full"  # All data


@dataclass
class UserContext:
    """User security context for data isolation."""
    user_id: int
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    department_id: Optional[str] = None
    team_ids: List[str] = None
    role: str = "user"
    permission_level: PermissionLevel = PermissionLevel.OWN
    
    # Row-level security attributes
    allowed_user_ids: Optional[Set[int]] = None
    allowed_tenant_ids: Optional[Set[str]] = None
    
    # Custom filters
    custom_filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.team_ids is None:
            self.team_ids = []
        if self.custom_filters is None:
            self.custom_filters = {}


@dataclass
class TableRLS:
    """Row-level security configuration for a table."""
    table_name: str
    user_column: Optional[str] = None  # Column matching user_id
    tenant_column: Optional[str] = None  # Column matching tenant_id
    org_column: Optional[str] = None  # Column matching organization_id
    department_column: Optional[str] = None
    team_column: Optional[str] = None  # Array or single team ID
    
    # Soft delete column
    deleted_at_column: Optional[str] = "deleted_at"
    
    # Additional predicates (AND conditions)
    custom_predicates: List[str] = None
    
    def __post_init__(self):
        if self.custom_predicates is None:
            self.custom_predicates = []


class PermissionAwareSQLGenerator:
    """Generate SQL with row-level security predicates."""
    
    # Common RLS configurations (can be extended via config)
    DEFAULT_RLS_CONFIGS = {
        "users": TableRLS(
            table_name="users",
            user_column="id",
            tenant_column="tenant_id",
        ),
        "orders": TableRLS(
            table_name="orders",
            user_column="user_id",
            tenant_column="tenant_id",
            org_column="organization_id",
        ),
        "customers": TableRLS(
            table_name="customers",
            user_column="owner_id",
            tenant_column="tenant_id",
            org_column="org_id",
        ),
        "documents": TableRLS(
            table_name="documents",
            user_column="created_by",
            tenant_column="tenant_id",
            department_column="department_id",
        ),
        "projects": TableRLS(
            table_name="projects",
            user_column="owner_id",
            tenant_column="tenant_id",
            team_column="team_ids",  # Array column
        ),
    }
    
    def __init__(self, rls_configs: Optional[Dict[str, TableRLS]] = None):
        """Initialize with RLS configurations."""
        self.rls_configs = rls_configs or self.DEFAULT_RLS_CONFIGS.copy()
    
    def apply_rls_to_query(
        self,
        sql: str,
        user_context: UserContext,
        extracted_tables: Optional[List[str]] = None,
    ) -> str:
        """Apply row-level security to a SQL query.
        
        Args:
            sql: Original SQL query
            user_context: User's security context
            extracted_tables: Tables detected in query (optional)
            
        Returns:
            Modified SQL with security predicates
        """
        if user_context.permission_level == PermissionLevel.FULL:
            return sql  # No filtering needed
        
        # Extract tables if not provided
        if not extracted_tables:
            extracted_tables = self._extract_tables_from_sql(sql)
        
        # Build security predicates for each table
        modified_sql = sql
        
        for table_name in extracted_tables:
            rls = self.rls_configs.get(table_name)
            if not rls:
                continue
            
            predicates = self._build_predicates(rls, user_context)
            if predicates:
                modified_sql = self._inject_predicate(modified_sql, table_name, predicates)
        
        return modified_sql
    
    def _build_predicates(self, rls: TableRLS, ctx: UserContext) -> List[str]:
        """Build WHERE predicates for a table based on user context."""
        predicates = []
        
        # Soft delete check (always applied)
        if rls.deleted_at_column:
            predicates.append(f"{rls.deleted_at_column} IS NULL")
        
        if ctx.permission_level == PermissionLevel.OWN:
            # Only own data
            if rls.user_column:
                predicates.append(f"{rls.user_column} = {ctx.user_id}")
            
        elif ctx.permission_level == PermissionLevel.TEAM:
            # Own data OR team data
            user_predicates = []
            
            if rls.user_column:
                user_predicates.append(f"{rls.user_column} = {ctx.user_id}")
            
            if rls.team_column and ctx.team_ids:
                if rls.team_column.endswith("s"):  # Likely array
                    # PostgreSQL array containment
                    teams_str = ",".join(f"'{t}'" for t in ctx.team_ids)
                    user_predicates.append(
                        f"{rls.team_column} && ARRAY[{teams_str}]"
                    )
                else:
                    teams_str = ",".join(f"'{t}'" for t in ctx.team_ids)
                    user_predicates.append(f"{rls.team_column} IN ({teams_str})")
            
            if user_predicates:
                predicates.append(f"({' OR '.join(user_predicates)})")
                
        elif ctx.permission_level == PermissionLevel.DEPARTMENT:
            # Department level
            if rls.department_column and ctx.department_id:
                predicates.append(f"{rls.department_column} = '{ctx.department_id}'")
            elif rls.user_column:
                # Fall back to user-level if no department column
                predicates.append(f"{rls.user_column} = {ctx.user_id}")
        
        # Tenant isolation (applied regardless of permission level)
        if ctx.tenant_id and rls.tenant_column:
            predicates.append(f"{rls.tenant_column} = '{ctx.tenant_id}'")
        
        # Organization isolation
        if ctx.organization_id and rls.org_column:
            predicates.append(f"{rls.org_column} = '{ctx.organization_id}'")
        
        # Custom predicates
        for custom in rls.custom_predicates:
            predicates.append(custom)
        
        return predicates
    
    def _inject_predicate(
        self,
        sql: str,
        table_name: str,
        predicates: List[str],
    ) -> str:
        """Inject security predicates into SQL query."""
        if not predicates:
            return sql
        
        security_clause = " AND ".join(predicates)
        
        # Pattern to find the table in FROM or JOIN clause
        # This is a simplified approach - production might need a proper SQL parser
        
        # Try to inject after WHERE clause for this table
        # Simple heuristic: add a CTE with security filter
        
        # Create a secure version using CTE
        secure_table_name = f"secure_{table_name}"
        
        cte = f"""
WITH {secure_table_name} AS (
    SELECT * FROM {table_name}
    WHERE {security_clause}
)
""".strip()
        
        # Replace table references with secure CTE
        # This is simplified - real implementation would use a SQL parser
        modified_sql = sql.replace(f"FROM {table_name}", f"FROM {secure_table_name}")
        modified_sql = modified_sql.replace(f"JOIN {table_name}", f"JOIN {secure_table_name}")
        
        # Add CTE at the beginning
        if modified_sql != sql:
            if sql.upper().startswith("SELECT"):
                modified_sql = cte + "\n" + modified_sql
        
        return modified_sql
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = set()
        
        # Simple regex patterns (production should use a proper parser)
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'DELETE\s+FROM\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(matches)
        
        return list(tables)
    
    def get_visible_context_description(
        self,
        user_context: UserContext,
        available_tables: List[str],
    ) -> str:
        """Generate a description of what data the user can see.
        
        This is useful for the LLM to understand data boundaries.
        """
        parts = ["DATA ACCESS BOUNDARIES:"]
        
        if user_context.permission_level == PermissionLevel.FULL:
            parts.append("- You have FULL access to all data in the database.")
            return "\n".join(parts)
        
        parts.append(f"- Permission Level: {user_context.permission_level.value}")
        parts.append(f"- Your User ID: {user_context.user_id}")
        
        if user_context.tenant_id:
            parts.append(f"- Tenant ID: {user_context.tenant_id} (data isolation active)")
        
        if user_context.organization_id:
            parts.append(f"- Organization ID: {user_context.organization_id}")
        
        if user_context.department_id:
            parts.append(f"- Department ID: {user_context.department_id}")
        
        if user_context.team_ids:
            parts.append(f"- Team IDs: {', '.join(user_context.team_ids)}")
        
        # Add table-specific info
        table_access = []
        for table in available_tables:
            rls = self.rls_configs.get(table)
            if rls:
                if user_context.permission_level == PermissionLevel.OWN:
                    table_access.append(f"  - {table}: Only rows where {rls.user_column} = {user_context.user_id}")
                elif user_context.permission_level == PermissionLevel.TEAM:
                    access_desc = f"{table}: Your own rows OR team rows"
                    table_access.append(f"  - {access_desc}")
                else:
                    table_access.append(f"  - {table}: All accessible rows")
        
        if table_access:
            parts.append("\nTable Access Rules:")
            parts.extend(table_access)
        
        parts.append("\nIMPORTANT: All queries will be automatically filtered by these rules.")
        
        return "\n".join(parts)


class QuerySanitizer:
    """Sanitize and validate queries for security."""
    
    # Dangerous operations to block
    BLOCKED_PATTERNS = [
        (r'\bDROP\s+', "DROP operations not allowed"),
        (r'\bTRUNCATE\s+', "TRUNCATE not allowed"),
        (r'\bALTER\s+', "ALTER not allowed"),
        (r'\bCREATE\s+(?:TABLE|INDEX|VIEW)', "CREATE not allowed"),
        (r'\bGRANT\s+', "GRANT not allowed"),
        (r'\bREVOKE\s+', "REVOKE not allowed"),
        (r';\s*\w+', "Multiple statements not allowed"),
    ]
    
    # Allowed operations for read-only
    ALLOWED_OPERATIONS = ['SELECT', 'WITH', 'EXPLAIN']
    
    def sanitize(self, sql: str) -> tuple[str, Optional[str]]:
        """Sanitize a SQL query.
        
        Returns:
            (sanitized_sql, error_message)
            If error_message is not None, the query should be rejected.
        """
        # Check for blocked patterns
        sql_upper = sql.upper()
        
        for pattern, message in self.BLOCKED_PATTERNS:
            if re.search(pattern, sql_upper):
                return sql, message
        
        # Verify it's a read-only query
        first_keyword = None
        for keyword in self.ALLOWED_OPERATIONS:
            if sql_upper.strip().startswith(keyword):
                first_keyword = keyword
                break
        
        if not first_keyword:
            return sql, "Only SELECT queries are allowed"
        
        # Clean the query
        sanitized = sql.strip()
        
        # Remove trailing semicolons (prevents injection)
        sanitized = sanitized.rstrip(';')
        
        # Remove comments
        sanitized = re.sub(r'--[^\n]*', '', sanitized)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        return sanitized, None


# Global instance
_permission_sql_generator: Optional[PermissionAwareSQLGenerator] = None
_query_sanitizer: Optional[QuerySanitizer] = None


def get_permission_sql_generator() -> PermissionAwareSQLGenerator:
    """Get the global permission-aware SQL generator."""
    global _permission_sql_generator
    if _permission_sql_generator is None:
        _permission_sql_generator = PermissionAwareSQLGenerator()
    return _permission_sql_generator


def get_query_sanitizer() -> QuerySanitizer:
    """Get the global query sanitizer."""
    global _query_sanitizer
    if _query_sanitizer is None:
        _query_sanitizer = QuerySanitizer()
    return _query_sanitizer


def create_user_context(
    user_id: int,
    tenant_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    permission_level: PermissionLevel = PermissionLevel.OWN,
    **kwargs
) -> UserContext:
    """Helper to create a user context."""
    return UserContext(
        user_id=user_id,
        tenant_id=tenant_id,
        organization_id=organization_id,
        permission_level=permission_level,
        **kwargs
    )
