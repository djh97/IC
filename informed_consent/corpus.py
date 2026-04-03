from __future__ import annotations

from collections import Counter
from html import unescape
from math import log
from pathlib import Path
from typing import Any
import json
import re

from .types import ChunkRecord, ConsentSourceDocument, RetrievalHit, SourceTextUnit

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency at import time
    PdfReader = None


SUPPORTED_SOURCE_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".html",
    ".htm",
    ".pdf",
}


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
HTML_META_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "title": [
        re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+name=["\']dcterms\.title["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL),
    ],
    "description": [
        re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+name=["\']dcterms\.description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    ],
}
HTML_BLOCK_TAG_PATTERN = re.compile(r"</?(?:p|div|section|article|main|header|footer|nav|aside|ul|ol|li|br|tr|table|h[1-6])[^>]*>", re.IGNORECASE)
HTML_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^skip to ", re.IGNORECASE),
    re.compile(r"^menu$", re.IGNORECASE),
    re.compile(r"^search$", re.IGNORECASE),
    re.compile(r"^footer$", re.IGNORECASE),
    re.compile(r"^home$", re.IGNORECASE),
    re.compile(r"^breadcrumb", re.IGNORECASE),
    re.compile(r"^return to ", re.IGNORECASE),
    re.compile(r"^back to top$", re.IGNORECASE),
    re.compile(r"^contact us$", re.IGNORECASE),
]


def stringify_sequence(values: list[Any]) -> str:
    return ", ".join(str(item).strip() for item in values if str(item).strip())


def clean_markup_text(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    text = text.replace("\\[", "[").replace("\\]", "]")
    return normalize_text(text)


def get_nested(payload: dict[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def slugify_source_id(relative_path: Path) -> str:
    stem = relative_path.with_suffix("").as_posix().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_") or "source"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def estimate_token_count(text: str) -> int:
    return len(tokenize(text))


def build_citation_label(title: str, metadata: dict[str, object]) -> str:
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return f"{title}, p. {page_number}"
    return title


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_html_metadata(html: str, fallback_title: str) -> dict[str, str]:
    metadata: dict[str, str] = {"title": fallback_title}
    for field_name, patterns in HTML_META_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(html)
            if match:
                metadata[field_name] = normalize_text(match.group(1))
                break
    return metadata


def prune_html_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    previous = None
    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            continue
        if any(pattern.search(line) for pattern in HTML_BOILERPLATE_LINE_PATTERNS):
            continue
        if line == previous:
            continue
        filtered.append(line)
        previous = line
    return filtered


def trim_html_lines_to_content(lines: list[str], anchor_title: str | None) -> list[str]:
    trimmed = lines
    if anchor_title:
        anchor_lower = anchor_title.lower().strip()
        for index, line in enumerate(lines):
            if anchor_lower and anchor_lower in line.lower():
                trimmed = lines[max(index - 2, 0):]
                break

    footer_markers = {"feedback", "about fda", "accessibility", "contact fda", "fda archive"}
    for index, line in enumerate(trimmed):
        if line.lower() in footer_markers and index > 10:
            return trimmed[:index]
    return trimmed


def html_to_text(html: str, *, anchor_title: str | None = None) -> str:
    text = re.sub(r"<head.*?</head>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<noscript.*?</noscript>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<svg.*?</svg>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = HTML_BLOCK_TAG_PATTERN.sub("\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = prune_html_lines(text.splitlines())
    lines = trim_html_lines_to_content(lines, anchor_title=anchor_title)
    return normalize_text("\n".join(lines))


def is_clinicaltrials_study_record(payload: dict[str, object]) -> bool:
    nct_id = get_nested(payload, "protocolSection", "identificationModule", "nctId")
    return isinstance(nct_id, str) and nct_id.upper().startswith("NCT")


def build_study_source_title(payload: dict[str, object], fallback_title: str) -> str:
    identification = get_nested(payload, "protocolSection", "identificationModule")
    if not isinstance(identification, dict):
        return fallback_title
    brief_title = clean_markup_text(str(identification.get("briefTitle") or ""))
    official_title = clean_markup_text(str(identification.get("officialTitle") or ""))
    nct_id = str(identification.get("nctId") or "").strip().upper()
    title = brief_title or official_title or fallback_title
    if nct_id and nct_id not in title:
        return f"{title} ({nct_id})"
    return title


def build_clinicaltrials_source_units(source_doc: ConsentSourceDocument, payload: dict[str, object]) -> list[SourceTextUnit]:
    protocol_section = get_nested(payload, "protocolSection")
    if not isinstance(protocol_section, dict):
        return []

    identification = get_nested(protocol_section, "identificationModule")
    status_module = get_nested(protocol_section, "statusModule")
    sponsor_module = get_nested(protocol_section, "sponsorCollaboratorsModule")
    description_module = get_nested(protocol_section, "descriptionModule")
    conditions_module = get_nested(protocol_section, "conditionsModule")
    design_module = get_nested(protocol_section, "designModule")
    arms_module = get_nested(protocol_section, "armsInterventionsModule")
    outcomes_module = get_nested(protocol_section, "outcomesModule")
    eligibility_module = get_nested(protocol_section, "eligibilityModule")
    contacts_module = get_nested(protocol_section, "contactsLocationsModule")
    more_info_module = get_nested(protocol_section, "moreInfoModule")
    document_section = get_nested(payload, "documentSection")
    derived_section = get_nested(payload, "derivedSection")

    identification = identification if isinstance(identification, dict) else {}
    status_module = status_module if isinstance(status_module, dict) else {}
    sponsor_module = sponsor_module if isinstance(sponsor_module, dict) else {}
    description_module = description_module if isinstance(description_module, dict) else {}
    conditions_module = conditions_module if isinstance(conditions_module, dict) else {}
    design_module = design_module if isinstance(design_module, dict) else {}
    arms_module = arms_module if isinstance(arms_module, dict) else {}
    outcomes_module = outcomes_module if isinstance(outcomes_module, dict) else {}
    eligibility_module = eligibility_module if isinstance(eligibility_module, dict) else {}
    contacts_module = contacts_module if isinstance(contacts_module, dict) else {}
    more_info_module = more_info_module if isinstance(more_info_module, dict) else {}
    document_section = document_section if isinstance(document_section, dict) else {}
    derived_section = derived_section if isinstance(derived_section, dict) else {}

    nct_id = str(identification.get("nctId") or source_doc.source_id).strip().upper()
    source_title = build_study_source_title(payload, source_doc.title)

    lead_sponsor = ""
    if isinstance(sponsor_module.get("leadSponsor"), dict):
        lead_sponsor = clean_markup_text(str(sponsor_module["leadSponsor"].get("name") or ""))

    conditions = conditions_module.get("conditions")
    conditions = [clean_markup_text(str(item)) for item in conditions] if isinstance(conditions, list) else []
    keywords = conditions_module.get("keywords")
    keywords = [clean_markup_text(str(item)) for item in keywords] if isinstance(keywords, list) else []
    phases = design_module.get("phases")
    phases = [str(item).strip() for item in phases] if isinstance(phases, list) else []

    enrollment_info = design_module.get("enrollmentInfo")
    enrollment_text = ""
    if isinstance(enrollment_info, dict):
        count = enrollment_info.get("count")
        count_type = str(enrollment_info.get("type") or "").strip().lower()
        if count is not None:
            enrollment_text = f"{count} participants"
            if count_type:
                enrollment_text += f" ({count_type})"

    design_info = design_module.get("designInfo")
    design_info = design_info if isinstance(design_info, dict) else {}
    arms = arms_module.get("armGroups")
    arms = arms if isinstance(arms, list) else []
    interventions = arms_module.get("interventions")
    interventions = interventions if isinstance(interventions, list) else []
    primary_outcomes = outcomes_module.get("primaryOutcomes")
    primary_outcomes = primary_outcomes if isinstance(primary_outcomes, list) else []
    secondary_outcomes = outcomes_module.get("secondaryOutcomes")
    secondary_outcomes = secondary_outcomes if isinstance(secondary_outcomes, list) else []
    locations = contacts_module.get("locations")
    locations = locations if isinstance(locations, list) else []
    point_of_contact = more_info_module.get("pointOfContact")
    point_of_contact = point_of_contact if isinstance(point_of_contact, dict) else {}
    large_docs = get_nested(document_section, "largeDocumentModule", "largeDocs")
    large_docs = large_docs if isinstance(large_docs, list) else []

    location_countries = sorted(
        {
            clean_markup_text(str(location.get("country") or ""))
            for location in locations
            if isinstance(location, dict) and str(location.get("country") or "").strip()
        }
    )
    location_examples = []
    for location in locations[:5]:
        if not isinstance(location, dict):
            continue
        city = clean_markup_text(str(location.get("city") or ""))
        state = clean_markup_text(str(location.get("state") or ""))
        country = clean_markup_text(str(location.get("country") or ""))
        example = ", ".join(part for part in [city, state, country] if part)
        if example:
            location_examples.append(example)

    sections: list[tuple[str, list[str]]] = [
        (
            "overview",
            [
                f"Study record: {source_title}",
                f"NCT ID: {nct_id}",
                f"Brief title: {clean_markup_text(str(identification.get('briefTitle') or ''))}",
                f"Official title: {clean_markup_text(str(identification.get('officialTitle') or ''))}",
                f"Sponsor: {lead_sponsor}",
                f"Overall status: {clean_markup_text(str(status_module.get('overallStatus') or ''))}",
                f"Study type: {clean_markup_text(str(design_module.get('studyType') or ''))}",
                f"Phase: {stringify_sequence(phases)}",
                f"Enrollment: {enrollment_text}",
                f"Conditions: {stringify_sequence(conditions)}",
                f"Keywords: {stringify_sequence(keywords)}",
                f"Brief summary: {clean_markup_text(str(description_module.get('briefSummary') or ''))}",
            ],
        ),
        (
            "design_and_interventions",
            [
                f"Primary purpose: {clean_markup_text(str(design_info.get('primaryPurpose') or ''))}",
                f"Allocation: {clean_markup_text(str(design_info.get('allocation') or ''))}",
                f"Intervention model: {clean_markup_text(str(design_info.get('interventionModel') or ''))}",
                f"Observational model: {clean_markup_text(str(design_info.get('observationalModel') or ''))}",
                f"Time perspective: {clean_markup_text(str(design_info.get('timePerspective') or ''))}",
                f"Target duration: {clean_markup_text(str(design_module.get('targetDuration') or ''))}",
                "Study arms: "
                + " | ".join(
                    clean_markup_text(
                        f"{arm.get('label')}: {arm.get('description') or arm.get('type') or ''}"
                    )
                    for arm in arms
                    if isinstance(arm, dict) and str(arm.get("label") or "").strip()
                ),
                "Interventions: "
                + " | ".join(
                    clean_markup_text(
                        f"{item.get('type')}: {item.get('name')}. {item.get('description') or ''}"
                    )
                    for item in interventions
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ),
                f"Detailed description: {clean_markup_text(str(description_module.get('detailedDescription') or ''))}",
            ],
        ),
        (
            "eligibility_and_participation",
            [
                f"Sex eligible: {clean_markup_text(str(eligibility_module.get('sex') or ''))}",
                f"Minimum age: {clean_markup_text(str(eligibility_module.get('minimumAge') or ''))}",
                f"Maximum age: {clean_markup_text(str(eligibility_module.get('maximumAge') or ''))}",
                f"Standard ages: {stringify_sequence(list(eligibility_module.get('stdAges') or []))}",
                f"Healthy volunteers: {clean_markup_text(str(eligibility_module.get('healthyVolunteers') or ''))}",
                f"Study population: {clean_markup_text(str(eligibility_module.get('studyPopulation') or ''))}",
                f"Eligibility criteria: {clean_markup_text(str(eligibility_module.get('eligibilityCriteria') or ''))}",
            ],
        ),
        (
            "outcomes_and_timeline",
            [
                "Primary outcomes: "
                + " | ".join(
                    clean_markup_text(
                        f"{item.get('measure')}. {item.get('description') or ''} Time frame: {item.get('timeFrame') or ''}"
                    )
                    for item in primary_outcomes
                    if isinstance(item, dict) and str(item.get("measure") or "").strip()
                ),
                "Secondary outcomes: "
                + " | ".join(
                    clean_markup_text(
                        f"{item.get('measure')}. {item.get('description') or ''} Time frame: {item.get('timeFrame') or ''}"
                    )
                    for item in secondary_outcomes
                    if isinstance(item, dict) and str(item.get("measure") or "").strip()
                ),
                f"Study start date: {clean_markup_text(str(get_nested(status_module, 'startDateStruct', 'date') or ''))}",
                f"Primary completion date: {clean_markup_text(str(get_nested(status_module, 'primaryCompletionDateStruct', 'date') or ''))}",
                f"Study completion date: {clean_markup_text(str(get_nested(status_module, 'completionDateStruct', 'date') or ''))}",
                f"Results posted: {clean_markup_text(str(payload.get('hasResults') or ''))}",
            ],
        ),
        (
            "contacts_documents_and_results",
            [
                f"Point of contact: {clean_markup_text(str(point_of_contact.get('title') or ''))}",
                f"Point of contact organization: {clean_markup_text(str(point_of_contact.get('organization') or ''))}",
                f"Point of contact email: {clean_markup_text(str(point_of_contact.get('email') or ''))}",
                f"Location count: {len(locations)}",
                f"Location countries: {stringify_sequence(location_countries)}",
                f"Example locations: {' | '.join(location_examples)}",
                "Study documents: "
                + " | ".join(
                    clean_markup_text(
                        f"{item.get('label')} (protocol={item.get('hasProtocol')}, sap={item.get('hasSap')}, icf={item.get('hasIcf')})"
                    )
                    for item in large_docs
                    if isinstance(item, dict) and str(item.get("label") or "").strip()
                ),
                f"Results available: {clean_markup_text(str(payload.get('hasResults') or ''))}",
                f"Record version holder: {clean_markup_text(str(get_nested(derived_section, 'miscInfoModule', 'versionHolder') or ''))}",
            ],
        ),
    ]

    units: list[SourceTextUnit] = []
    for index, (section_name, lines) in enumerate(sections, start=1):
        cleaned_lines = [line.strip() for line in lines if isinstance(line, str) and line.strip()]
        if not cleaned_lines:
            continue
        text = normalize_text("\n".join(cleaned_lines))
        if not text:
            continue
        unit_metadata = {
            **source_doc.metadata,
            "extraction_method": "clinicaltrials_json_v2",
            "source_title": source_title,
            "nct_id": nct_id,
            "section_name": section_name,
            "overall_status": clean_markup_text(str(status_module.get("overallStatus") or "")),
            "study_type": clean_markup_text(str(design_module.get("studyType") or "")),
            "has_results": bool(payload.get("hasResults")),
        }
        units.append(
            SourceTextUnit(
                unit_id=f"{source_doc.source_id}:section:{index}",
                source_id=source_doc.source_id,
                text=text,
                char_count=len(text),
                token_count_estimate=estimate_token_count(text),
                citation_label=f"{source_title}, {section_name.replace('_', ' ')}",
                metadata=unit_metadata,
            )
        )
    return units


def load_source_text_units(source_doc: ConsentSourceDocument) -> list[SourceTextUnit]:
    source_path = Path(source_doc.path)
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        return []

    if suffix == ".pdf":
        return load_pdf_text_units(source_doc)

    raw_text = source_path.read_text(encoding="utf-8", errors="ignore")

    source_title = source_doc.title
    html_metadata: dict[str, str] | None = None

    if suffix in {".json"}:
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict) and is_clinicaltrials_study_record(payload):
                return build_clinicaltrials_source_units(source_doc, payload)
            text = json.dumps(payload, indent=2, ensure_ascii=True)
        except json.JSONDecodeError:
            text = raw_text
    elif suffix in {".html", ".htm"}:
        html_metadata = extract_html_metadata(raw_text, source_doc.title)
        text = html_to_text(raw_text, anchor_title=html_metadata.get("title"))
        source_title = html_metadata.get("title", source_doc.title)
    else:
        text = normalize_text(raw_text)

    if not text:
        return []

    unit_metadata = {
        **source_doc.metadata,
        "extraction_method": "plain_text",
        "source_title": source_title,
    }
    if html_metadata:
        unit_metadata["extraction_method"] = "html_text"
        unit_metadata["html_title"] = source_title
        if html_metadata.get("description"):
            unit_metadata["html_description"] = html_metadata["description"]

    return [
        SourceTextUnit(
            unit_id=f"{source_doc.source_id}:unit:1",
            source_id=source_doc.source_id,
            text=text,
            char_count=len(text),
            token_count_estimate=estimate_token_count(text),
            citation_label=build_citation_label(source_title, unit_metadata),
            metadata=unit_metadata,
        )
    ]


def load_pdf_text_units(source_doc: ConsentSourceDocument) -> list[SourceTextUnit]:
    if PdfReader is None:
        raise RuntimeError(
            "PDF support requires pypdf. Install the dependencies from requirements.txt before loading PDF sources."
        )

    reader = PdfReader(source_doc.path)
    units: list[SourceTextUnit] = []
    for page_index, page in enumerate(reader.pages, start=1):
        extracted = normalize_text(page.extract_text() or "")
        if not extracted:
            continue
        unit_metadata = {
            **source_doc.metadata,
            "page_number": page_index,
            "extraction_method": "pypdf",
            "source_title": source_doc.title,
        }
        units.append(
            SourceTextUnit(
                unit_id=f"{source_doc.source_id}:page:{page_index}",
                source_id=source_doc.source_id,
                text=extracted,
                char_count=len(extracted),
                token_count_estimate=estimate_token_count(extracted),
                citation_label=build_citation_label(source_doc.title, unit_metadata),
                metadata=unit_metadata,
            )
        )
    return units


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + chunk_size, len(normalized))
        end = hard_end

        if hard_end < len(normalized):
            search_start = max(start + int(chunk_size * 0.6), start + 1)
            breakpoints = [
                normalized.rfind(". ", search_start, hard_end),
                normalized.rfind("; ", search_start, hard_end),
                normalized.rfind(": ", search_start, hard_end),
                normalized.rfind(" ", search_start, hard_end),
            ]
            best = max(breakpoints)
            if best > start:
                end = best + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def build_chunk_records(
    text_units: list[SourceTextUnit],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for unit in text_units:
        unit_chunks = split_text_with_overlap(unit.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, chunk_text in enumerate(unit_chunks, start=1):
            metadata = {
                **unit.metadata,
                "unit_id": unit.unit_id,
                "chunk_index_within_unit": chunk_index,
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{unit.unit_id}:chunk:{chunk_index}",
                    source_id=unit.source_id,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count_estimate=estimate_token_count(chunk_text),
                    citation_label=unit.citation_label,
                    metadata=metadata,
                )
            )
    return chunks


def load_chunk_records(path: Path) -> list[ChunkRecord]:
    rows: list[ChunkRecord] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(ChunkRecord(**json.loads(line)))
    return rows


def retrieve_lexical_hits(chunks: list[ChunkRecord], query: str, top_k: int = 5) -> list[RetrievalHit]:
    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []

    chunk_tokens = [tokenize(chunk.text) for chunk in chunks]
    doc_freq: Counter[str] = Counter()
    for terms in chunk_tokens:
        for term in set(terms):
            doc_freq[term] += 1

    avg_doc_len = sum(len(terms) for terms in chunk_tokens) / max(len(chunk_tokens), 1)
    scored_hits: list[tuple[float, ChunkRecord]] = []
    k1 = 1.5
    b = 0.75
    total_docs = len(chunk_tokens)

    for chunk, terms in zip(chunks, chunk_tokens, strict=True):
        term_freq = Counter(terms)
        doc_len = len(terms) or 1
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
            score += idf * (numerator / denominator)
        if score > 0:
            scored_hits.append((score, chunk))

    scored_hits.sort(key=lambda item: item[0], reverse=True)

    hits: list[RetrievalHit] = []
    for rank, (score, chunk) in enumerate(scored_hits[:top_k], start=1):
        excerpt = chunk.text[:400].strip()
        hits.append(
            RetrievalHit(
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                rank=rank,
                score=round(score, 6),
                citation_label=chunk.citation_label,
                excerpt=excerpt,
                metadata=chunk.metadata,
            )
        )
    return hits
