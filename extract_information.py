import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_client import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_FILE,
    DEFAULT_MODEL,
    RemoteChatClient,
    load_api_key_from_python_file,
)


def read_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def extract_json_object(s: str) -> Optional[str]:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1:
        return s[start : end + 1]
    return None


def extract_json_array(s: str) -> Optional[str]:
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1:
        return s[start : end + 1]
    return None


class MockExtractorClient:
    def complete(self, prompt: str) -> str:
        if "Please extract both the intent" in prompt:
            question = prompt.rsplit("Question:", 1)[-1].rsplit("Output:", 1)[0].strip()
            return json.dumps(mock_extract_entity_intent(question), ensure_ascii=False)
        if "Extract evidence relations" in prompt:
            entities = []
            if "Entities:" in prompt:
                raw_entities = prompt.rsplit("Entities:", 1)[-1].rsplit("Output:", 1)[0].strip()
                try:
                    entities = json.loads(raw_entities)
                except json.JSONDecodeError:
                    entities = []
            return json.dumps(mock_extract_relation(entities), ensure_ascii=False)
        return "{}"


def mock_extract_entity_intent(question: str) -> Dict[str, Any]:
    lower = question.lower().strip()
    quoted = re.findall(r'"([^"]+)"|' + r"'([^']+)'", question)
    entities = [left or right for left, right in quoted]

    day_match = re.match(r"what day is (.+?)\??$", lower)
    if day_match:
        entities.append(day_match.group(1).strip().title())

    count_match = re.match(r"how many ([a-z ]+?) (?:are|is) in (.+?)\??$", lower)
    if count_match:
        entities.append(count_match.group(2).strip().title())
        entities.append(count_match.group(1).strip())

    capital_phrases = re.findall(r"\b(?:[A-Z][A-Za-z0-9.&'-]*\s*){1,5}", question)
    stopwords = {
        "are",
        "is",
        "what",
        "which",
        "who",
        "where",
        "when",
        "how",
        "many",
        "same",
        "the",
    }
    for phrase in capital_phrases:
        phrase = phrase.strip(" ,?.")
        if phrase and phrase.lower() not in stopwords:
            entities.append(phrase)

    numbers = re.findall(r"\b\d{2,4}(?:\.\d+)?\b", question)
    entities.extend(numbers)

    if not entities:
        words = [word.strip(" ,?.") for word in question.split() if len(word.strip(" ,?.")) > 3]
        entities = words[:3]

    deduped = []
    seen = set()
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(entity)

    intent = "Answer support information"
    if any(word in lower for word in ["city", "county", "located", "where"]):
        intent = "Location information"
    elif any(word in lower for word in ["date", "day", "when", "year"]):
        intent = "Date or time information"
    elif any(word in lower for word in ["how many", "number", "count"]):
        intent = "Numeric answer information"
    elif any(word in lower for word in ["who", "writer", "singer", "author"]):
        intent = "Person identity information"

    return {"Intent": intent, "Entities": deduped[:6]}


def mock_extract_relation(entities: List[str]) -> List[Dict[str, Any]]:
    if len(entities) < 2:
        return []
    return [{"Entities": entities[:2], "Description": "is related to"}]


def extract_entity_intent_short(question: str, llm: Any) -> Dict[str, Any]:
    prompt = (
        "Please extract both the intent and evidence nodes of the question, using the following criteria:\n\n"
        "1) For intent, indicate the content intent of the evidence that the question expects, without specific details.\n"
        "2) For evidence nodes, extract the specific details of the question.\n\n"
        "The output must be JSON with keys Intent and Entities.\n"
        'Example: Question: What nationality was James Henry Miller\'s wife?\n'
        'Output: {"Intent": "Nationality of person", "Entities": ["James Henry Miller", "wife"]}\n\n'
        f"Question: {question} Output:"
    )
    response = llm.complete(prompt)
    clean = extract_json_object(response)
    if clean is None:
        raise ValueError(f"Could not parse entity/intent JSON from response: {response}")
    parsed = json.loads(clean)
    if "Entities" not in parsed and "evidence nodes" in parsed:
        parsed["Entities"] = parsed.pop("evidence nodes")
    parsed.setdefault("Intent", "")
    parsed.setdefault("Entities", [])
    return parsed


def extract_relation(question: str, entities: List[str], llm: Any) -> List[Dict[str, Any]]:
    prompt = (
        "Extract evidence relations from the input question and evidence nodes. Requirements:\n"
        "1) Each relation contains two fields: Entities and Description.\n"
        "2) Relation descriptions only involve the connected nodes.\n"
        "3) Return [] if no relation exists between nodes.\n\n"
        'Example: Q: Lee Jun-fan played what character in "The Green Hornet" television series?\n'
        'Nodes: ["Lee Jun-fan", "The Green Hornet"]\n'
        'Out: [{"Entities":["Lee Jun-fan","The Green Hornet"], "Description": "played character in"}]\n\n'
        f"Question: {question}\nEntities: {json.dumps(entities, ensure_ascii=False)} Output:"
    )
    response = llm.complete(prompt)
    clean = extract_json_array(response)
    if clean is None:
        raise ValueError(f"Could not parse relation JSON from response: {response}")
    relations = json.loads(clean)
    normalized = []
    for relation in relations:
        normalized.append(
            {
                "Entities": relation.get("Entities") or relation.get("Evidence nodes") or [],
                "Description": relation.get("Description") or relation.get("Evidence Relations") or "",
            }
        )
    return normalized


def iter_records(raw_data: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_data, list):
        return [dict(item) for item in raw_data]
    if isinstance(raw_data, dict):
        records = []
        for key, value in raw_data.items():
            record = dict(value)
            record.setdefault("id", key)
            records.append(record)
        return records
    raise TypeError("Input JSON must be a list or an object keyed by record id.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AuthChain intent, evidence nodes, and relations.")
    parser.add_argument("--input", default="data/reproduction/sample_questions.json", help="Input QA JSON.")
    parser.add_argument("--output", default="data/reproduction/sample_extract_info.json", help="Output JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process.")
    parser.add_argument("--offset", type=int, default=0, help="Number of records to skip before processing.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for live extraction.")
    parser.add_argument(
        "--api-base-url",
        "--base-url",
        dest="api_base_url",
        default=DEFAULT_API_BASE_URL,
        help="AntChat/OpenAI-compatible base URL.",
    )
    parser.add_argument("--api-key-env", default="ANTCHAT_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE, help="Python file containing an API_KEY constant.")
    parser.add_argument("--api-key-var", default="api_key", help="Variable name to read from --api-key-file.")
    parser.add_argument("--timeout", type=int, default=60, help="Single request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for one model call.")
    parser.add_argument("--request-sleep", type=float, default=0.0, help="Seconds to sleep after each successful request.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP proxy settings from environment.")
    parser.add_argument("--mock", action="store_true", help="Run deterministic offline smoke test without OpenAI calls.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing extract_information fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace it.")

    raw_data = read_json(args.input)
    records = iter_records(raw_data)
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    if args.mock:
        llm = MockExtractorClient()
    else:
        api_key = os.getenv(args.api_key_env, "") or load_api_key_from_python_file(args.api_key_file, args.api_key_var)
        llm = RemoteChatClient(
            api_base_url=args.api_base_url,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            request_sleep=args.request_sleep,
            use_env_proxy=args.use_env_proxy,
            enable_thinking=False,
        )

    outputs = []
    for num, record in enumerate(selected, start=args.offset):
        print(f"processing {num}: {record.get('id', '<no-id>')}")
        if args.skip_existing and record.get("extract_information"):
            outputs.append(record)
            continue
        question = record["question"]
        info = extract_entity_intent_short(question, llm)
        info["Relations"] = extract_relation(question, info["Entities"], llm)
        updated = dict(record)
        updated["extract_information"] = info
        outputs.append(updated)

    write_json(outputs, args.output)
    print(f"done: wrote {len(outputs)} records to {args.output}")


if __name__ == "__main__":
    main()
