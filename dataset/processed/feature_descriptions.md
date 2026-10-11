# URL Feature Descriptions

This document provides a comprehensive dictionary of all features extracted for the **Multi-Layer Phishing Detection & Risk Assessment** system.

All features are strictly numeric (`int64` or `float64`) and derived without requiring internet connectivity or web page content execution.

---

## 1. Length & Structural Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `url_length` | Integer | Total length of the URL in characters. | Phishing URLs often conceal destinations via deep subdirectory chains, obfuscated parameters, or encoded payloads, resulting in elevated length. |
| `hostname_length` | Integer | Length of the parsed hostname / FQDN. | Attackers frequently embed brand names or high-entropy random character strings into subdomains and hosts. |
| `path_length` | Integer | Length of the URL path component. | Fraudulent login clones often reside deep in hijacked server directories (e.g., `/wp-content/plugins/.../login.php`). |
| `query_length` | Integer | Length of the query string following `?`. | Phishing campaigns frequently pass tracking tokens, victim emails, and redirect targets via query strings. |
| `num_dots` | Integer | Total count of dot (`.`) characters in the full URL. | Phishers often create multi-level subdomains or mimic domains using periods in paths/queries. |
| `num_hyphens` | Integer | Total count of hyphen (`-`) characters. | Brand spoofing frequently uses hyphens to simulate authentic domains (e.g., `paypal-security-update.com`). |
| `num_underscores` | Integer | Total count of underscore (`_`) characters. | Less common in genuine domain names, commonly seen in auto-generated phishing script paths. |
| `num_slashes` | Integer | Total count of forward slash (`/`) characters. | Indicates nesting depth of paths and directories on the hosting server. |
| `num_question_marks`| Integer | Total count of question mark (`?`) characters. | Demarcates query strings; multiple question marks often signal parameter obfuscation. |
| `num_equal_signs` | Integer | Total count of equal sign (`=`) characters. | Reflects the density of query parameter key-value pairs. |
| `num_ampersands` | Integer | Total count of ampersand (`&`) characters. | Delimits multiple parameters in complex phishing tracking schemes. |
| `num_at_symbols` | Integer | Total count of `@` symbols in the URL. | The `@` symbol causes browsers to ignore all preceding text in the authority component, treating the subsequent host as the actual target. |
| `num_percent_symbols` | Integer | Total count of percent (`%`) characters. | Percent signs indicate URL hexadecimal encoding, often used to bypass basic keyword filters. |
| `num_digits` | Integer | Total count of numeric digits (`0`-`9`) across the URL. | High digit density is characteristic of randomized hostnames, IP addresses, and tracking identifiers. |
| `num_special_characters`| Integer | Total count of non-alphanumeric characters in the URL. | Quantifies overall symbol complexity and character variance. |

---

## 2. Hostname & Domain Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `has_ip_address` | Binary (0/1) | Indicates whether the hostname is an IPv4 or IPv6 address. | Legitimate commercial services rely on registered domain names. IP address hostnames are a strong indicator of malicious infrastructure. |
| `hostname_has_digits` | Binary (0/1) | 1 if the hostname contains numeric digits, 0 otherwise. | Algorithmic domain generation (DGA) and cheap VPS hosting often introduce numbers into hostnames. |
| `hostname_has_hyphen` | Binary (0/1) | 1 if the hostname contains hyphens, 0 otherwise. | Hyphenated hostnames are a standard technique for domain squatting (typosquatting / combosquatting). |
| `num_subdomains` | Integer | Estimated count of subdomains using Public Suffix List rules. | Excessive subdomain layers are used to construct convincing lures on free hosting platforms (e.g., `login.secure.bank.pages.dev`). |
| `has_multiple_subdomains` | Binary (0/1) | 1 if `num_subdomains` > 1, 0 otherwise. | Flags nested subdomain hierarchies commonly exploited by adversaries. |
| `is_https` | Binary (0/1) | 1 if URL protocol is `https://`, 0 otherwise. | While modern phishing also adopts free SSL certs, protocol distribution remains informative when correlated with other structural signals. |
| `has_explicit_port` | Binary (0/1) | 1 if an explicit non-default port is designated in authority, 0 otherwise. | Standard web traffic uses default ports; explicit ports often indicate compromised rogue servers. |
| `has_unusual_port` | Binary (0/1) | 1 if port is specified and outside standard HTTP(80) / HTTPS(443), 0 otherwise. | Non-standard ports (e.g., `:8080`, `:8443`, `:8888`) are common on temporary attacker testing nodes. |
| `num_query_parameters`| Integer | Count of parsed key-value pairs in the query string. | Quantifies dynamic parameter density used in targeted credential harvesting. |

---

## 3. Suspicious-Pattern Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `has_url_shortener` | Binary (0/1) | 1 if hostname matches known URL shorteners (bit.ly, t.co, etc.), 0 otherwise. | URL shorteners conceal the true destination and prevent immediate user inspection. |
| `has_suspicious_keyword` | Binary (0/1) | 1 if URL contains sensitive lure words (`login`, `verify`, `account`, `password`, etc.), 0 otherwise. | Social engineering lures concentrate heavily around authentication and account-action verbs. |
| `has_encoded_characters` | Binary (0/1) | 1 if URL contains percent-encoded hex sequences (`%`), 0 otherwise. | Hex-encoding is used to mask executable script names or deceptive host fragments. |
| `has_at_symbol` | Binary (0/1) | 1 if URL contains `@` symbol, 0 otherwise. | Flags credential injection or userinfo spoofing in URL authority. |

---

## Target Label Specification

- **`0`**: **Legitimate** (Verified authentic, benign web resources)
- **`1`**: **Phishing** (Deceptive, malicious, or credential-harvesting web resources)
