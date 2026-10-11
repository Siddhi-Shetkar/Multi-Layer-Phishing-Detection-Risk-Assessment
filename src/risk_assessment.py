"""
Module for URL Risk Assessment and Explainable Security Threat Evaluation.
Multi-Layer Phishing Detection & Risk Assessment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from src.feature_extraction import FEATURE_COLUMNS


# Default, configurable threshold boundaries for risk categorization
DEFAULT_LOW_THRESHOLD: float = 0.35
DEFAULT_HIGH_THRESHOLD: float = 0.70


class RiskAssessmentEngine:
    """
    Configurable multi-layer risk assessment engine.

    Combines statistical machine learning output probabilities with empirical
    URL structural security indicators to classify URLs into LOW, MEDIUM, or HIGH risk,
    accompanied by transparent, evidence-based threat explanations.
    """

    def __init__(
        self,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD
    ):
        """
        Initialize the Risk Assessment Engine with configurable thresholds.

        Args:
            low_threshold: Maximum probability for LOW risk (default: 0.35).
                           Probabilities strictly below this value are considered LOW,
                           provided no severe heuristic red flags are triggered.
            high_threshold: Minimum probability for HIGH risk (default: 0.70).
                            Probabilities at or above this value are categorized as HIGH risk.
                            Probabilities in [low_threshold, high_threshold) are MEDIUM risk.
        """
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: low_threshold ({low_threshold}) must be < "
                f"high_threshold ({high_threshold}) and both within [0.0, 1.0]."
            )
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)

    def analyze_url_features(self, features: Dict[str, Union[int, float]]) -> Tuple[List[str], List[str]]:
        """
        Analyze extracted features to generate human-readable security signals.

        Only analyzes features present in FEATURE_COLUMNS.
        Does not make claims that HTTPS alone guarantees safety or maliciousness.

        Args:
            features: Dictionary of extracted URL features adhering to FEATURE_COLUMNS.

        Returns:
            Tuple of (warnings_list, safe_signals_list).
        """
        warnings: List[str] = []
        safe_signals: List[str] = []

        # 1. Host & Infrastructure
        if features.get('has_ip_address', 0) == 1:
            warnings.append(
                "Raw IP Address Hostname: URL hosts destination directly on an IP address instead of a domain name, "
                "bypassing standard DNS registration."
            )
        else:
            safe_signals.append("Uses registered domain name (no raw IP host detected).")

        if features.get('has_url_shortener', 0) == 1:
            warnings.append(
                "URL Shortening Service Detected: Hostname matches a known shortener (e.g., bit.ly, tinyurl), "
                "masking the true final landing domain."
            )

        if features.get('has_unusual_port', 0) == 1:
            warnings.append(
                "Non-Standard Port: The URL explicitly specifies a non-standard web port (outside standard 80/443)."
            )

        # 2. Credential harvesting keywords
        if features.get('has_suspicious_keyword', 0) == 1:
            warnings.append(
                "Sensitive Authentication Lure: Contains sensitive action keywords commonly targeted in phishing lures "
                "(e.g., 'login', 'verify', 'update', 'account', 'password')."
            )
        else:
            safe_signals.append("No sensitive authentication or credential-harvesting lure keywords identified.")

        # 3. Authority manipulation & Obfuscation
        if features.get('has_at_symbol', 0) == 1:
            warnings.append(
                "Authority Override Symbol ('@'): Presence of '@' can cause browsers to ignore the preceding prefix "
                "and navigate to the subsequent host, a classic brand-spoofing technique."
            )

        if features.get('has_encoded_characters', 0) == 1:
            warnings.append(
                f"Hexadecimal Percent Encoding: Found {features.get('num_percent_symbols', 0)} percent symbol(s), "
                "which may obfuscate characters to evade simple pattern filters."
            )

        # 4. Domain & Subdomain complexity
        num_subdomains = features.get('num_subdomains', 0)
        if num_subdomains > 2:
            warnings.append(
                f"Excessive Subdomain Nesting: Detected {num_subdomains} subdomain levels. Attackers often nest subdomains "
                "to mimic trustworthy entities (e.g., 'brand.verify.account.attacker.com')."
            )
        elif num_subdomains <= 1 and features.get('has_ip_address', 0) == 0:
            safe_signals.append("Clean domain structure with standard subdomain depth.")

        if features.get('hostname_has_hyphen', 0) == 1:
            warnings.append(
                "Hyphenated Hostname: Host contains hyphens, a technique commonly associated with domain combosquatting."
            )

        # 5. Length & Parameter Density
        url_len = features.get('url_length', 0)
        if url_len > 100:
            warnings.append(
                f"Elevated URL Length ({url_len} chars): Significantly longer than typical web links, "
                "frequently observed in phishing redirect chains."
            )

        query_params = features.get('num_query_parameters', 0)
        if query_params >= 3:
            warnings.append(
                f"Dense Query Parameter Density: Contains {query_params} dynamic parameters, often used to pass victim tokens "
                "or tracking parameters."
            )

        # 6. Protocol Note (Balanced & objective)
        is_https = features.get('is_https', 0)
        if is_https == 0:
            warnings.append(
                "Unencrypted HTTP Protocol: URL transmits traffic unencrypted (HTTP). Legitimate sensitive portals require HTTPS."
            )
        else:
            safe_signals.append(
                "Uses HTTPS encrypted transport (Note: Modern phishing attacks may also acquire free SSL certificates)."
            )

        return warnings, safe_signals

    def assess_risk(
        self,
        url: str,
        features: Dict[str, Union[int, float]],
        prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive risk assessment given a URL, its features, and model predictions.

        Risk Categories:
        - LOW: Probability < low_threshold and no severe deceptive indicators.
        - MEDIUM: Probability in [low_threshold, high_threshold), or low probability with multiple suspicious anomalies.
        - HIGH: Probability >= high_threshold, or model predicts PHISHING with strong indicators.

        Args:
            url: The input URL string.
            features: Extracted numeric features.
            prediction_result: Output from model prediction (predicted_class, phishing_probability, confidence).

        Returns:
            Structured risk assessment dictionary.
        """
        prob = float(prediction_result.get('phishing_probability', 0.5))
        pred_class = int(prediction_result.get('predicted_class', 1 if prob >= 0.5 else 0))
        confidence = float(prediction_result.get('confidence', max(prob, 1.0 - prob)))

        warnings, safe_signals = self.analyze_url_features(features)

        # Severity flags that escalate minimum risk
        critical_flags = []
        if features.get('has_ip_address', 0) == 1:
            critical_flags.append("Direct IP address hosting")
        if features.get('has_at_symbol', 0) == 1:
            critical_flags.append("Authority '@' symbol spoofing")
        if features.get('has_url_shortener', 0) == 1:
            critical_flags.append("Concealed destination via URL shortener")

        # Initial risk assignment strictly based on configurable model probability thresholds
        if prob >= self.high_threshold:
            risk_level = "HIGH"
            reason_summary = (
                f"Model estimated high phishing probability ({prob * 100:.1f}%), exceeding the high-risk threshold "
                f"({self.high_threshold * 100:.0f}%)."
            )
        elif prob >= self.low_threshold:
            risk_level = "MEDIUM"
            reason_summary = (
                f"Model estimated borderline/uncertain phishing probability ({prob * 100:.1f}%), falling in the "
                f"caution range ({self.low_threshold * 100:.0f}% - {self.high_threshold * 100:.0f}%)."
            )
        else:
            risk_level = "LOW"
            reason_summary = (
                f"Model estimated low phishing probability ({prob * 100:.1f}%), within the acceptable threshold "
                f"(< {self.low_threshold * 100:.0f}%)."
            )

        # Heuristic escalation: If model probability was LOW but critical red flags are present,
        # escalate to MEDIUM to alert user to manual verification necessity.
        escalated = False
        if risk_level == "LOW" and (len(critical_flags) > 0 or len(warnings) >= 3):
            risk_level = "MEDIUM"
            escalated = True
            reason_summary += (
                f" Escalated to MEDIUM risk due to {len(critical_flags)} critical security anomaly: "
                f"{', '.join(critical_flags)}."
            )

        # Calculate a combined risk score (0 to 100 scale)
        # Base score from probability + bonus points from active structural warnings
        base_score = prob * 100.0
        warning_penalty = min(20.0, len(warnings) * 4.0)
        risk_score = round(min(100.0, max(0.0, (base_score * 0.8) + (warning_penalty))), 1)

        disclaimer = (
            "DISCLAIMER: This assessment is generated via statistical Machine Learning classification "
            "and lexical URL heuristics. It serves as an advisory decision-support signal and does not "
            "guarantee absolute safety or malice. Never enter credentials into unverified domains."
        )

        structured_result = {
            "url": url,
            "predicted_class": pred_class,
            "classification": "PHISHING" if pred_class == 1 else "LEGITIMATE",
            "risk_level": risk_level,
            "risk_score": risk_score,
            "phishing_probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "thresholds": {
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
            },
            "reason_summary": reason_summary,
            "escalated_by_heuristics": escalated,
            "critical_flags": critical_flags,
            "warnings": warnings,
            "safe_signals": safe_signals,
            "disclaimer": disclaimer,
        }

        return structured_result


def assess_url_risk(
    url: str,
    features: Dict[str, Union[int, float]],
    prediction_result: Dict[str, Any],
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD
) -> Dict[str, Any]:
    """
    Convenience function for standalone risk assessment.

    Args:
        url: Raw URL string.
        features: Extracted features dictionary.
        prediction_result: Model prediction output dictionary.
        low_threshold: Threshold below which risk is LOW.
        high_threshold: Threshold above which risk is HIGH.

    Returns:
        Structured risk assessment dictionary.
    """
    engine = RiskAssessmentEngine(low_threshold=low_threshold, high_threshold=high_threshold)
    return engine.assess_risk(url, features, prediction_result)
