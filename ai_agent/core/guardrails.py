"""Guardrails and safety checks for Agent IQ.

This module provides:
- PII detection and redaction
- Prompt injection detection
- Content moderation
- SQL injection prevention
- Tool permission validation
"""
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


class GuardrailResult:
    """Result of a guardrail check."""
    
    def __init__(
        self,
        passed: bool,
        violation_type: Optional[str] = None,
        message: Optional[str] = None,
        severity: str = "medium",  # low, medium, high, critical
        action: str = "warn",  # allow, warn, block, require_approval
        details: Optional[Dict] = None,
    ):
        self.passed = passed
        self.violation_type = violation_type
        self.message = message
        self.severity = severity
        self.action = action
        self.details = details or {}


class GuardrailType(str, Enum):
    """Types of guardrails."""
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    CONTENT_MODERATION = "content_moderation"
    SQL_INJECTION = "sql_injection"
    TOOL_PERMISSION = "tool_permission"
    RATE_LIMIT = "rate_limit"
    COST_LIMIT = "cost_limit"


class GuardrailsService:
    """Service for applying safety guardrails."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|delete|drop|alter|create|truncate|exec|execute)\b)",
        r"(--|#|/\*|\*/)",  # Comments
        r"(\b(or|and)\b\s+\d+\s*=\s*\d+)",  # Boolean-based
        r"(sleep\s*\(|benchmark\s*\()",  # Time-based
        r"(load_file\s*\(|into\s+outfile)",  # File operations
    ]

    # PII patterns (basic regex-based detection)
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
        "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "api_key": r"\b(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}['\"]?",
    }

    # Whitelist patterns for business identifiers that should not be flagged as PII
    BUSINESS_ID_PATTERNS = [
        r"\b[A-Z]{2}\d{8,}\b",  # Sales order IDs like SO10000018
        r"\b[A-Z]{3}\d{6,}\b",  # Other business IDs
    ]

    # Prompt injection indicators
    PROMPT_INJECTION_INDICATORS = [
        "ignore previous instructions",
        "ignore the above",
        "disregard",
        "forget everything",
        "you are now",
        "new instructions",
        "system prompt",
        "ignore your programming",
        "do not follow",
        "override",
        "jailbreak",
        "DAN",
        "developer mode",
    ]

    def __init__(self):
        """Initialize the guardrails service."""
        self._presidio_available = False
        
        # Try to import Presidio for better PII detection
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
            })
            self._presidio_analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
            self._presidio_available = True
            log.info("Presidio PII analyzer loaded")
        except ImportError:
            log.info("Presidio not available, using regex-based PII detection")

    def check_input(
        self,
        text: str,
        user_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        check_types: Optional[List[GuardrailType]] = None,
    ) -> List[GuardrailResult]:
        """Check user input for safety violations.

        Args:
            text: Input text to check
            user_id: User ID for context
            agent_id: Agent ID for context
            check_types: Specific guardrail types to check (default: all)

        Returns:
            List of guardrail results
        """
        from ai_agent.core.config import settings

        results = []
        check_types = check_types or list(GuardrailType)

        for check_type in check_types:
            # Skip PII check if disabled in config
            if check_type == GuardrailType.PII and not settings.ENABLE_PII_DETECTION:
                continue

            # Skip prompt injection check if disabled in config
            if check_type == GuardrailType.PROMPT_INJECTION and not settings.ENABLE_PROMPT_INJECTION_DETECTION:
                continue

            # Skip content moderation check if disabled in config
            if check_type == GuardrailType.CONTENT_MODERATION and not settings.ENABLE_CONTENT_MODERATION:
                continue

            if check_type == GuardrailType.PII:
                result = self._check_pii(text)
                if not result.passed:
                    results.append(result)

            elif check_type == GuardrailType.PROMPT_INJECTION:
                result = self._check_prompt_injection(text)
                if not result.passed:
                    results.append(result)

            elif check_type == GuardrailType.CONTENT_MODERATION:
                result = self._check_content_moderation(text)
                if not result.passed:
                    results.append(result)

            elif check_type == GuardrailType.SQL_INJECTION:
                result = self._check_sql_injection(text)
                if not result.passed:
                    results.append(result)

        return results

    def check_output(
        self,
        text: str,
        user_id: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> List[GuardrailResult]:
        """Check AI output for safety violations.
        
        Args:
            text: Output text to check
            user_id: User ID for context
            agent_id: Agent ID for context
            
        Returns:
            List of guardrail results
        """
        results = []
        
        # Check for PII leakage
        pii_result = self._check_pii(text)
        if not pii_result.passed:
            results.append(pii_result)
        
        # Check for content policy violations
        mod_result = self._check_content_moderation(text)
        if not mod_result.passed:
            results.append(mod_result)
        
        return results

    def check_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: int,
        agent_id: str,
        tool_permissions: Optional[Dict] = None,
    ) -> GuardrailResult:
        """Check if a tool call is permitted.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            user_id: User ID
            agent_id: Agent ID
            tool_permissions: Tool permission configuration
            
        Returns:
            Guardrail result
        """
        # Check tool permission
        if tool_permissions:
            permission = tool_permissions.get(tool_name, {})
            
            # Check if tool is allowed
            if permission.get("blocked", False):
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailType.TOOL_PERMISSION,
                    message=f"Tool '{tool_name}' is blocked for this agent",
                    severity="high",
                    action="block",
                )
            
            # Check if approval is required
            if permission.get("requires_approval", False):
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailType.TOOL_PERMISSION,
                    message=f"Tool '{tool_name}' requires human approval",
                    severity="medium",
                    action="require_approval",
                )
            
            # Check destructive operations
            if permission.get("destructive", False):
                # Require explicit confirmation
                return GuardrailResult(
                    passed=True,  # Allow but flag
                    violation_type=GuardrailType.TOOL_PERMISSION,
                    message=f"Tool '{tool_name}' performs destructive operations",
                    severity="medium",
                    action="warn",
                )
        
        # Check for SQL injection in tool arguments
        for key, value in arguments.items():
            if isinstance(value, str):
                sql_check = self._check_sql_injection(value)
                if not sql_check.passed:
                    return GuardrailResult(
                        passed=False,
                        violation_type=GuardrailType.SQL_INJECTION,
                        message=f"Potential SQL injection in tool argument '{key}'",
                        severity="critical",
                        action="block",
                        details={"argument": key, "value": value[:100]},
                    )
        
        return GuardrailResult(passed=True)

    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Redact PII from text.
        
        Args:
            text: Text to redact
            replacement: Replacement string
            
        Returns:
            Redacted text
        """
        redacted = text
        
        # Use Presidio if available
        if self._presidio_available:
            try:
                results = self._presidio_analyzer.analyze(text=text, language="en")
                # Sort by start position in reverse to avoid offset issues
                for result in sorted(results, key=lambda x: x.start, reverse=True):
                    redacted = redacted[:result.start] + replacement + redacted[result.end:]
                return redacted
            except Exception as e:
                log.warning(f"Presidio redaction failed: {e}")
        
        # Fallback to regex
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        
        return redacted

    def _check_pii(self, text: str) -> GuardrailResult:
        """Check for PII in text."""
        import re

        # First check if the text contains whitelisted business IDs
        for pattern in self.BUSINESS_ID_PATTERNS:
            if re.search(pattern, text):
                log.info(f"Text contains whitelisted business ID pattern, skipping PII check")
                return GuardrailResult(passed=True)

        detected_types = []

        # Check regex patterns
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                if pii_type not in detected_types:
                    detected_types.append(pii_type)

        # Check with Presidio if available
        if self._presidio_available:
            try:
                from presidio_analyzer import PatternRecognizer

                results = self._presidio_analyzer.analyze(
                    text=text,
                    language="en",
                )

                for result in results:
                    # Skip Presidio detections for whitelisted business IDs
                    if result.entity_type.lower() in ["id", "identifier", "order_id"]:
                        continue
                    if result.entity_type not in detected_types:
                        detected_types.append(result.entity_type)

            except Exception as e:
                log.error(f"Presidio analysis failed: {e}")

        if detected_types:
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailType.PII,
                message=f"Potential PII detected: {', '.join(detected_types)}",
                severity="medium",
            )

        return GuardrailResult(passed=True)

    def _check_prompt_injection(self, text: str) -> GuardrailResult:
        """Check for prompt injection attempts."""
        text_lower = text.lower()
        detected_indicators = []
        
        for indicator in self.PROMPT_INJECTION_INDICATORS:
            if indicator in text_lower:
                detected_indicators.append(indicator)
        
        # Check for suspicious characters/patterns
        suspicious_patterns = [
            r"<\|",  # Special tokens
            r"\[INST\]",  # Instruction tokens
            r"<<SYS>>",  # System tokens
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_indicators.append(f"pattern: {pattern}")
        
        if detected_indicators:
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailType.PROMPT_INJECTION,
                message="Potential prompt injection detected",
                severity="critical",
                action="block",
                details={"indicators": detected_indicators[:5]},  # Limit details
            )
        
        return GuardrailResult(passed=True)

    def _check_content_moderation(self, text: str) -> GuardrailResult:
        """Check for content policy violations."""
        # Basic checks - in production, integrate with moderation APIs
        # (OpenAI Moderation, Azure Content Safety, etc.)
        
        # Check for obvious harmful content indicators
        harmful_indicators = [
            "kill yourself",
            "self-harm",
            "terrorist",
            "bomb making",
            "child abuse",
        ]
        
        text_lower = text.lower()
        detected = []
        
        for indicator in harmful_indicators:
            if indicator in text_lower:
                detected.append(indicator)
        
        if detected:
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailType.CONTENT_MODERATION,
                message="Content violates safety policies",
                severity="critical",
                action="block",
                details={"violations": detected},
            )
        
        return GuardrailResult(passed=True)

    def _check_sql_injection(self, text: str) -> GuardrailResult:
        """Check for SQL injection patterns."""
        text_upper = text.upper()
        
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper):
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailType.SQL_INJECTION,
                    message="Potential SQL injection detected",
                    severity="critical",
                    action="block",
                )
        
        return GuardrailResult(passed=True)

    def get_remediation(
        self,
        violation: GuardrailResult,
        context: Optional[Dict] = None,
    ) -> str:
        """Get remediation message for a violation.
        
        Args:
            violation: The guardrail violation
            context: Additional context
            
        Returns:
            User-friendly remediation message
        """
        if violation.violation_type == GuardrailType.PII:
            return "I've detected potentially sensitive information in your message. Please avoid sharing personal details like emails, phone numbers, or ID numbers."
        
        elif violation.violation_type == GuardrailType.PROMPT_INJECTION:
            return "I'm unable to process this request as it appears to contain instructions that could interfere with my operation."
        
        elif violation.violation_type == GuardrailType.CONTENT_MODERATION:
            return "I'm unable to assist with this request as it may violate content safety policies."
        
        elif violation.violation_type == GuardrailType.SQL_INJECTION:
            return "I've detected potentially unsafe database commands. Please ensure your queries only use SELECT statements for data retrieval."
        
        elif violation.violation_type == GuardrailType.TOOL_PERMISSION:
            if violation.action == "require_approval":
                return f"This action requires approval. Please wait for confirmation before proceeding."
            return "This action is not permitted with the current configuration."
        
        return "I'm unable to process this request due to safety constraints."


def get_guardrails_service() -> GuardrailsService:
    """Get the guardrails service singleton."""
    global _guardrails_service
    try:
        return _guardrails_service
    except NameError:
        _guardrails_service = GuardrailsService()
        return _guardrails_service
