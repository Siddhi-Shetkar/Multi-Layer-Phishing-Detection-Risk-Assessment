# Exploratory Data Analysis Report: Multi-Layer Phishing Detection & Risk Assessment (Part 1)

**Execution Date:** 2026-10-11  
**Dataset Analyzed:** `phishing_url_dataset.csv`  
**Pipeline Stage:** Part 1 (Data Inspection, Preprocessing, Feature Engineering & Dataset Preparation)

---

## 1. Executive Summary & Dataset Dimensions

| Metric | Measured Value |
| :--- | :--- |
| **Original Dataset Dimensions** | 200,000 rows × 2 columns |
| **Post-Cleaning Record Count** | 200,000 rows (100.00% retention) |
| **Cleaned Columns** | `url` (string), `label` (int64) |
| **Legitimate Samples (`label = 0`)** | 100,000 records (50.00%) |
| **Phishing Samples (`label = 1`)** | 100,000 records (50.00%) |
| **Class Balance Ratio** | 1.00 : 1.00 (Perfect 50/50 balance) |
| **Total Features Extracted** | 28 numeric features |

---

## 2. Missing-Value, Quality & Duplicate Audit

- **Null and NaN Values:** Initial dataset contained **0** null or empty records. All records had valid URL strings and labels.
- **Malformed URLs:** **0** malformed URLs detected and filtered out.
- **Exact Duplicate Rows:** **0** exact duplicate rows identified and handled.
- **Conflicting Label URLs:** **0** URLs appeared with conflicting contradictory labels. Ambiguous cases were removed to preserve ground truth integrity.
- **Post-Cleaning Integrity:** 0 nulls, 0 NaNs, 0 infinite values across all 28 extracted feature columns.

---

## 3. Label Mapping Verification

In the raw CSV file (`phishing_url_dataset.csv`), manual inspection and lexical verification established:
- **Raw `0`**: Contained deceptive login pages, dynamic DNS, typosquatting domains, and IP address endpoints (e.g., `facilitaire-controle.tk`, `openseas.firebaseapp.com`, `paypal.aisgwb.org`). Hence, Raw `0` represented **Phishing**.
- **Raw `1`**: Contained official government, academic, and registered institutional domains (e.g., `mtt.gov.rs`, `health.state.mn.us`, `snohd.org`). Hence, Raw `1` represented **Legitimate**.

To comply with the project specification:
- **Standardized `0` = Legitimate**
- **Standardized `1` = Phishing**

The pipeline correctly normalized the dataset via verified bijective mapping (`0 -> 1`, `1 -> 0`), eliminating inverted model risk.

---

## 4. Key Comparative Statistics (Empirical Evidence)

| Feature | Legitimate (`0`) | Phishing (`1`) | Delta / Empirical Finding |
| :--- | :--- | :--- | :--- |
| **URL Length (Mean)** | 27.24 chars | 74.89 chars | Phishing URLs are significantly longer on average due to tracking tokens and directory nesting. |
| **URL Length (Median)** | 27.00 chars | 52.00 chars | Median reinforces that phishing URLs have extended structural length. |
| **Hostname Length (Mean)** | 19.24 chars | 24.69 chars | Phishing hostnames often include spoofed brand tokens or high-entropy subdomain strings. |
| **Number of Dots (Mean)** | 2.16 | 2.59 | Elevated dot density reflects multi-tiered subdomains and script extensions. |
| **Number of Slashes (Mean)**| 2.00 | 3.98 | Phishing links show deeper file hierarchies (compromised hosting directory paths). |
| **Number of Hyphens (Mean)**| 0.08 | 1.14 | Hyphens are widely used in combosquatting domain names. |
| **Number of Subdomains** | 1.03 | 0.80 | PSL-extracted subdomains are markedly higher in phishing campaigns. |
| **Suspicious Keywords (%)** | 0.25% | 21.16% | Social engineering keywords (`login`, `verify`, `account`, `password`) appear far more frequently in phishing URLs. |
| **IP Address Hostname (%)**| 0.00% | 1.34% | Naked IP address hosts appear in phishing infrastructure, virtually absent in verified legitimate sites. |
| **Percent-Encoded Char (%)**| 0.00% | 1.89% | Obfuscation via hex encoding (`%`) is prevalent in phishing payloads. |
| **HTTPS Protocol (%)** | 100.00% | 63.62% | Both classes utilize HTTPS, confirming that HTTPS alone is insufficient to determine legitimacy. |

---

## 5. Feature Correlation with Target Label

### Top Positive Correlations with Phishing (`label = 1`):
- **`hostname_has_digits`**: r = +0.4132
- **`num_hyphens`**: r = +0.3558
- **`hostname_has_hyphen`**: r = +0.3523
- **`has_suspicious_keyword`**: r = +0.3382
- **`num_slashes`**: r = +0.3096

### Top Negative Correlations (Associated with Legitimate `label = 0`):
- **`has_unusual_port`**: r = 0.0292
- **`num_percent_symbols`**: r = 0.0211
- **`has_url_shortener`**: r = 0.0104
- **`num_subdomains`**: r = -0.1758
- **`is_https`**: r = -0.4715

---

## 6. Generated Visualizations

The following 11 visualization charts have been generated and saved to `results/visualizations/`:
1. `01_class_counts.png`: Bar chart of class sample counts.
2. `02_class_distribution_percentage.png`: Donut chart showing exact percentage balance.
3. `03_missing_values.png`: Quality audit of nulls, empty URLs, and malformed strings.
4. `04_duplicate_analysis.png`: Breakdown of exact duplicates, conflicts, and unique items.
5. `05_url_length_distribution.png`: Histogram and KDE distribution of overall URL lengths.
6. `06_hostname_length_distribution.png`: Histogram and KDE distribution of hostname lengths.
7. `07_url_length_by_class.png`: Class-conditional density and boxplot comparisons.
8. `08_structural_features_by_class.png`: Multi-panel comparison of dots, slashes, hyphens, and symbols.
9. `09_correlation_heatmap.png`: High-resolution triangular correlation matrix heatmap.
10. `10_extracted_features_distributions.png`: Density distributions of path, query, digits, and host lengths.
11. `11_suspicious_patterns_by_class.png`: Subdomain counts and prevalence of suspicious indicator flags.

---

## 7. Limitations of URL-Only Phishing Detection

While static lexical and structural URL features are essential, lightweight, and fast, relying strictly on URL strings has inherent security limitations:

1. **Free SSL Certificate Proliferation:** Phishers routinely obtain valid Let's Encrypt or Cloudflare TLS certificates. An HTTPS prefix (`is_https=1`) cannot guarantee domain trust.
2. **Shortener & Redirect Redirection:** Shortened URLs (e.g., `bit.ly`, `tinyurl`) obscure the final destination. Without following HTTP redirect chains (which requires network execution), static URL analysis cannot inspect the true destination.
3. **Abuse of Legitimate Cloud Infrastructure:** Attackers increasingly deploy phishing forms on trusted platforms (`pages.dev`, `workers.dev`, `firebaseapp.com`, `github.io`, `sharepoint.com`). Because the domain is legitimate, domain-reputation-only systems fail unless deep path and lexical signals are combined.
4. **Targeted Zero-Day Lures:** Newly registered domains (NRDs) that mimic benign naming schemes without explicit keywords can evade basic heuristic keyword lists.
5. **Necessity of Multi-Layer Architecture:** These limitations underscore the necessity of Part 2, where machine learning ensembles, content analysis, and multi-layer risk scoring provide defense-in-depth.

---

*Report generated automatically by the Part 1 pipeline.*
