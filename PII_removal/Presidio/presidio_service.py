from ast import pattern
from multiprocessing import context
from pydantic import BaseModel
from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import json, os, re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from presidio_analyzer.nlp_engine import NlpEngineProvider
import yaml


working_dir = os.path.dirname(os.path.abspath(__file__))
CUSTOM_RECOGNIZERS_FILE = os.path.join(working_dir, "custom_recognizers.yaml")
NOT_RECOGNIZERS_FILE = os.path.join(working_dir, "not_recognizers.yaml")
EXTERNAL_RECOGNIZERS_FILE = os.path.join(working_dir, "external_recognizers.yaml")
RECOGNIZER_FILES = [CUSTOM_RECOGNIZERS_FILE, NOT_RECOGNIZERS_FILE]
custom_recognizers = []
nlp_models = {}


# Global configuration for NLP engine
NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        {"lang_code": "de", "model_name": "de_core_news_lg"},
        {"lang_code": "fr", "model_name": "fr_core_news_lg"},
        {"lang_code": "es", "model_name": "es_core_news_lg"},
    ],
}

# Initialize engines
faker = Faker()
anonymizer = AnonymizerEngine()


# Pydantic models for input validation
class AnalyzeRequest(BaseModel):
    text: str


class AddRecognizerRequest(BaseModel):
    pattern_name: str
    regex: str


def fake_name(_=None):
    return faker.name()


def fake_time(_=None):
    return faker.time(pattern="%I:%M %p")


fake_operators = {
    "PERSON": OperatorConfig("custom", {"lambda": lambda x: faker.name()}),
    "PHONE_NUMBER": OperatorConfig(
        "custom", {"lambda": lambda x: faker.phone_number()}
    ),
    "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: faker.email()}),
    "TIME": OperatorConfig("custom", {"lambda": lambda x: fake_time()}),
}


@Language.factory("language_detector")
def create_lang_detector(nlp, name):
    return LanguageDetector()  # Remove seed parameter


def create_language_specific_recognizers(base_recognizer, languages):
    """Create separate recognizers for each language."""
    recognizers = []
    for lang in languages:
        # Create a copy of recognizer kwargs with language-specific name
        lang_recognizer_kwargs = {
            "name": f"{base_recognizer.name}_{lang}",  # Add language suffix
            "supported_entity": base_recognizer.supported_entities[0],
            "supported_language": lang,
        }

        # Add patterns if present
        if hasattr(base_recognizer, "patterns"):
            lang_recognizer_kwargs["patterns"] = base_recognizer.patterns

        # Add context if present
        if hasattr(base_recognizer, "context"):
            lang_recognizer_kwargs["context"] = base_recognizer.context

        # Add deny_list if present
        if hasattr(base_recognizer, "deny_list"):
            lang_recognizer_kwargs["deny_list"] = base_recognizer.deny_list

        # Create language-specific recognizer
        lang_recognizer = PatternRecognizer(**lang_recognizer_kwargs)
        recognizers.append(lang_recognizer)

    return recognizers


def add_pattern_recognizer(
    pattern_name: str, regex: str, context: list = [], language: str = "de"
):
    """Add a custom recognizer using regex pattern."""
    if isinstance(pattern_name, bytes):
        pattern_name = pattern_name.decode("utf-8")
    if isinstance(regex, bytes):
        regex = regex.decode("utf-8")
    if isinstance(language, bytes):
        language = language.decode("utf-8")

    # Process context list - handle both string and bytes, and split if needed
    processed_context = []
    if context:
        for ctx in context:
            if isinstance(ctx, bytes):
                ctx = ctx.decode("utf-8")
            # Remove b' prefix and ' suffix if present
            ctx = ctx.strip("b'").strip("'")
            if ctx:  # Only add non-empty strings
                processed_context.append(ctx)

    print(
        f"Adding pattern recognizer: \nName: {pattern_name}\nContext: {processed_context}\nLanguage: {language}"
    )

    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    # Create the pattern recognizer
    pattern = Pattern(name=pattern_name, regex=regex, score=0.9)

    recognizer_name = pattern_name + "_recognizer"
    counter = 2

    while any(
        recognizer["name"] == recognizer_name for recognizer in custom_recognizers
    ):
        recognizer_name = f"{pattern_name}_{counter}_recognizer"
        counter += 1

    # TODO:
    # check if recognizer name already exists / check if custom or builtin

    recognizer_kwargs = {
        "name": recognizer_name,
        "patterns": [pattern],
        "supported_entity": pattern_name.replace(" ", "_").upper(),
    }

    # Handle language support
    if language == "any":
        # Create base recognizer without language
        if processed_context != []:
            recognizer_kwargs["context"] = processed_context
        base_recognizer = PatternRecognizer(**recognizer_kwargs)
        # Create separate recognizers for each language
        language_recognizers = create_language_specific_recognizers(
            base_recognizer, ["en", "de", "fr", "es"]
        )

        # Add all language-specific recognizers to registry and save them
        for recognizer in language_recognizers:
            try:

                analyzer.registry.add_recognizer(recognizer)
                save_pattern_recognizer(recognizer)
            except Exception as e:
                print(f"Error adding {recognizer.name}: {str(e)}")
                continue
    else:
        # Single language case - add language suffix
        recognizer_kwargs["name"] = f"{recognizer_name}_{language}"
        recognizer_kwargs["supported_language"] = language
        if processed_context != []:
            recognizer_kwargs["context"] = processed_context
        custom_recognizer = PatternRecognizer(**recognizer_kwargs)
        analyzer.registry.add_recognizer(custom_recognizer)
        save_pattern_recognizer(custom_recognizer)

    return "SUCCESS"


def add_deny_list_recognizer(
    name: str, deny_list: list, context: list = [], language: str = "de"
):
    """Add a custom recognizer using deny list."""
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    if isinstance(language, bytes):
        language = language.decode("utf-8")
    processed_context = []
    processed_deny_list = []
    if context:
        for ctx in context:
            if isinstance(ctx, bytes):
                ctx = ctx.decode("utf-8")
            # Remove b' prefix and ' suffix if present
            ctx = ctx.strip("b'").strip("'")
            if ctx:  # Only add non-empty strings
                processed_context.append(ctx)
    if deny_list:
        for deny in deny_list:
            if isinstance(deny, bytes):
                deny = deny.decode("utf-8")
                deny = deny.strip("b'").strip("'")
            if deny:
                processed_deny_list.append(deny)

    print(
        f"Adding deny list recognizer: \nName: {name}\nContext: {processed_context}\nLanguage: {language}\nDeny list: {processed_deny_list}"
    )

    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    recognizer_name = name + "_DL-recognizer"
    recognizer_kwargs = {}

    recognizer_kwargs = {
        "name": recognizer_name,
        "deny_list": processed_deny_list,
        "supported_entity": name.replace(" ", "_").upper(),
    }

    if language != "any":
        recognizer_kwargs["supported_language"] = language
    else:
        recognizer_kwargs["supported_language"] = ["en", "de", "fr", "es"]

    if processed_context:
        recognizer_kwargs["context"] = processed_context

    if language == "any":
        # Create base recognizer without language
        base_recognizer = PatternRecognizer(**recognizer_kwargs)
        # Create separate recognizers for each language
        language_recognizers = create_language_specific_recognizers(
            base_recognizer, ["en", "de", "fr", "es"]
        )

        # Add all language-specific recognizers to registry and save them
        for recognizer in language_recognizers:
            try:
                analyzer.registry.add_recognizer(recognizer)
                save_deny_list_recognizer(recognizer)
            except Exception as e:
                print(f"Error adding {recognizer.name}: {str(e)}")
                continue
    else:
        # Single language case - add language suffix
        recognizer_kwargs["name"] = f"{recognizer_name}_{language}"
        recognizer_kwargs["supported_language"] = language
        custom_recognizer = PatternRecognizer(**recognizer_kwargs)
        analyzer.registry.add_recognizer(custom_recognizer)
        save_deny_list_recognizer(custom_recognizer)

    return "SUCCESS"


def load_nlp_models():
    # Register the language detector
    if not Language.has_factory("language_detector"):
        Language.factory("language_detector", func=create_lang_detector)

    for model in NLP_CONFIGURATION["models"]:
        lang_code = model["lang_code"]
        model_name = model["model_name"]
        try:
            nlp_models[lang_code] = spacy.load(model_name)
            # Add language detector to English model only
            if lang_code == "en":
                if "language_detector" not in nlp_models[lang_code].pipe_names:
                    try:
                        nlp_models[lang_code].add_pipe("language_detector", last=True)
                    except Exception as e:
                        print(f"Error adding language detector: {e}")
                        print("Continuing without language detection...")
        except OSError as e:
            print(f"Model {model_name} not found. Error: {e}")
            print(f"Please install it using: python -m spacy download {model_name}")


def get_language_model(text: str, specified_lang: str = "auto"):
    try:
        # If a specific language is provided and it's supported, use it directly
        if specified_lang != "auto":
            if specified_lang in nlp_models:
                print(f"Using specified language model: {specified_lang}")
                return nlp_models[specified_lang]
            else:
                print(
                    f"Specified language {specified_lang} not supported, falling back to detection"
                )

        # Only perform language detection if auto mode is selected
        doc = nlp_models["en"](text[:1000])  # Use first 1000 chars for detection
        detected = doc._.language
        lang_code = detected["language"]
        confidence = detected["score"]

        print(f"Detected language: {lang_code} (confidence: {confidence})")

        # Use detected language if supported and confidence > 0.8
        if lang_code in nlp_models and confidence > 0.8:
            return nlp_models[lang_code]

        print(f"Falling back to English model")
        return nlp_models["en"]

    except Exception as e:
        print(f"Language detection failed: {e}, using English model")
        return nlp_models["en"]


def create_analyzer_engine():
    """Create analyzer engine with all supported recognizers."""
    global analyzer, registry

    print("Creating analyzer engine...")

    # Ensure all YAML files exist with valid structure
    for yaml_file in RECOGNIZER_FILES:
        if not os.path.exists(yaml_file):
            with open(yaml_file, "w") as f:
                yaml.dump({"recognizers": []}, f, default_flow_style=False)
        else:
            # Ensure file has valid content
            try:
                with open(yaml_file, "r") as f:
                    content = yaml.safe_load(f)
                    if not content or "recognizers" not in content:
                        with open(yaml_file, "w") as f:
                            yaml.dump({"recognizers": []}, f, default_flow_style=False)
            except Exception:
                with open(yaml_file, "w") as f:
                    yaml.dump({"recognizers": []}, f, default_flow_style=False)

    supported_languages = ["en", "de", "fr", "es"]

    # Create NLP engine based on global defined configuration
    provider = NlpEngineProvider(nlp_configuration=NLP_CONFIGURATION)
    nlp_engine = provider.create_engine()

    # Create registry and load recognizers with supported languages
    registry = RecognizerRegistry()
    registry.supported_languages = supported_languages

    registry.load_predefined_recognizers()

    # Load custom recognizers from all YAML files
    for yaml_file in RECOGNIZER_FILES:
        if os.path.exists(yaml_file):
            registry.add_recognizers_from_yaml(yaml_file)

    # Create analyzer with updated registry
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=supported_languages,
    )

    print("Analyzer engine created with multi-language support.")


def analyze_text(text: str, active_labels: list = [], language: str = "auto"):
    try:
        # Ensure models are loaded
        if not nlp_models:
            load_nlp_models()

        if "analyzer" not in globals():
            create_analyzer_engine()
        else:
            load_custom_recognizers()

        if not active_labels:
            return []

        # Convert active_labels to list of strings if they're bytes or tuples
        processed_labels = []
        for label in active_labels:
            if isinstance(label, bytes):
                processed_labels.append(label.decode("utf-8"))
            elif isinstance(label, (list, tuple)):
                processed_labels.append(
                    label[0].decode("utf-8")
                    if isinstance(label[0], bytes)
                    else str(label[0])
                )
            else:
                processed_labels.append(str(label))

        print(f"Processed labels: {processed_labels}")

        if isinstance(text, bytes):
            text = text.decode("utf-8")

        if isinstance(language, bytes):
            language = language.decode("utf-8")

        # Get appropriate language model
        nlp = get_language_model(text, language)
        lang_code = nlp.lang if hasattr(nlp, "lang") else "en"
        print(f"Using language: {lang_code}")

        # Analyze
        all_results = analyzer.analyze(
            text=text,
            language=lang_code,
            score_threshold=0.3,
            return_decision_process=True,
        )

        # filter out false positives
        filtered_results = filter_false_positives(all_results, text)

        print(f"Filtered results: {filtered_results}")

        # Only include results for active labels
        active_label_filtered_results = []
        for result in filtered_results:
            if isinstance(result, tuple):
                _, _, _, entity_type, _ = result
                if entity_type in processed_labels:
                    active_label_filtered_results.append(result)
            else:
                if result.entity_type in processed_labels:
                    active_label_filtered_results.append(result)

        if not active_label_filtered_results:
            return []

        print(f"Active Label Filtered results: {active_label_filtered_results}")

        formatted_results = format_recognizer_results(
            text, active_label_filtered_results
        )
        # print(f"Analysis results: {formatted_results}")
        return formatted_results

    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        raise


def filter_false_positives(results, text):
    """Post-process to remove false positives"""
    filtered_results = []

    for result in results:
        # Check against deny patterns
        is_denied = any(
            r.entity_type.startswith("NOT_") and r.entity_type[4:] == result.entity_type
            for r in results
            if text[result.start : result.end].lower() == text[r.start : r.end].lower()
        )

        # Add only if not denied and meets score threshold
        if not is_denied and result.score >= 0.3:
            filtered_results.append(result)

    return filtered_results


# TODO: refactor
def analyze_file_and_anonymize(file_path: str):
    try:

        text, analyzer_results = analyze_file(file_path)

        converted_analyzer_results = convert_to_recognizer_results(analyzer_results)

        anonymized_results = anonymizer.anonymize(
            text=text, analyzer_results=converted_analyzer_results
        )
        print_colored_pii(anonymized_results.text)
        return "SEE_PRINT"

    except Exception as e:
        print(f"Error anonymizing file: {str(e)}")
        raise


# TODO: remove active_labels?
def anonymize_text(
    text: str,
    active_labels: list = [],
    language: str = "auto",
    get_analysis: bool = False,
):
    try:

        if isinstance(text, bytes):
            text = text.decode("utf-8")

        if isinstance(language, bytes):
            language = language.decode("utf-8")

        if isinstance(get_analysis, (bytes, int)):
            get_analysis = bool(get_analysis)

        print(f"get_analysis: {get_analysis}")

        # If no active labels, return the original text with empty results
        if not active_labels:
            return {"anonymized_text": text, "single_results": []}

        analyzer_results = analyze_text(text, active_labels, language)
        converted_analyzer_results = convert_to_recognizer_results(analyzer_results)

        # Get the anonymized text result
        anonymized_results = anonymizer.anonymize(
            text=text, analyzer_results=converted_analyzer_results
        )

        if get_analysis:
            analysis_data = []

            # Sort results by start position to ensure correct text extraction
            sorted_results = sorted(converted_analyzer_results, key=lambda x: x.start)
            anonymized_text = anonymized_results.text

            for result in sorted_results:
                # Extract the protected version from anonymized text
                protected_segment = f"<{result.entity_type}>"

                # Create analysis item
                item = {
                    "start": result.start,
                    "end": result.end,
                    "original_text": text[result.start : result.end],
                    "protected_text": protected_segment,
                    "recognizer_name": result.entity_type,
                    "score": result.score if hasattr(result, "score") else 0.85,
                    "pattern": result.score if hasattr(result, "score") else 0.85,
                }
                analysis_data.append(item)

            return {"anonymized_text": anonymized_text, "single_results": analysis_data}
        else:
            formatted_results = format_anonymizer_results(text, anonymized_results)
            print(
                {
                    "anonymized_text": anonymized_results.text,
                    "single_results": formatted_results,
                }
            )
            return {
                "anonymized_text": anonymized_results.text,
                "single_results": formatted_results,
            }

    except Exception as e:
        print(f"Error anonymizing text: {str(e)}")
        raise


# TODO: overlook
def pseudonymize_text(
    text: str,
    active_labels: list = [],
    language: str = "auto",
    get_analysis: bool = False,
):
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        if isinstance(language, bytes):
            language = language.decode("utf-8")

        if isinstance(get_analysis, (bytes, int)):
            get_analysis = bool(get_analysis)

        # If no active labels, return the original text with empty results
        if not active_labels:
            return {"anonymized_text": text, "single_results": []}

        analyzer_results = analyze_text(text, active_labels, language)

        converted_analyzer_results = convert_to_recognizer_results(analyzer_results)
        anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=converted_analyzer_results,
            operators=fake_operators,
        )

        if get_analysis:
            analysis_data = []
            for result, original_result in zip(
                converted_analyzer_results, analyzer_results
            ):
                item = {
                    "start": result.start,
                    "end": result.end,
                    "original_text": text[result.start : result.end] or "",
                    "protected_text": anonymized_results.text[result.start : result.end]
                    or "",
                    "recognizer_name": result.entity_type or "",
                    "score": result.score if hasattr(result, "score") else 0.85,
                    "pattern": original_result[4] if len(original_result) > 4 else 0.85,
                    "validation_result": None,  # Add validation logic if needed
                }
                analysis_data.append(item)

            return {
                "anonymized_text": anonymized_results.text,
                "single_results": analysis_data,
            }
        else:
            formatted_results = format_anonymizer_results(text, anonymized_results)
            return {
                "anonymized_text": anonymized_results.text,
                "single_results": formatted_results,
            }

    except Exception as e:
        print(f"Error pseudonymizing text: {str(e)}")
        raise


# TODO: convert
def analyze_file(file_path: str, language: str = "auto"):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            if "analyze_input" in data:
                text = data["analyze_input"]

                analyzer_results = analyze_text(text, language)

                return text, analyzer_results
            else:
                raise KeyError("Key 'analyze_input' not found in the JSON file.")
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        raise


def format_anonymizer_results(text, analyzer_results):
    try:
        # print(f"Debug - Input text: {text}")
        # print(f"Debug - Analyzer results: {analyzer_results}")

        formatted_results = []

        # Convert items() result to list if needed
        results_list = (
            analyzer_results.items
            if hasattr(analyzer_results, "items")
            else analyzer_results
        )

        for result in results_list:
            # Extract positions and text from result
            start = result["start"] if isinstance(result, dict) else result.start
            end = result["end"] if isinstance(result, dict) else result.end
            entity_type = (
                result["entity_type"]
                if isinstance(result, dict)
                else result.entity_type
            )
            anonymized = result["text"] if isinstance(result, dict) else result.text

            formatted_results.append(
                (
                    anonymized,  # Anonymized text
                    start,  # Start position
                    end,  # End position
                    entity_type,  # Entity type
                )
            )

            print(f"Debug - Formatted result: {formatted_results[-1]}")

        return formatted_results

    except Exception as e:
        print(f"Error in format_anonymizer_results: {str(e)}")
        return []


def format_recognizer_results(text, analyzer_results: list):
    formatted_results = []
    for result in analyzer_results:
        formatted_results.append(
            (
                text[result.start : result.end],
                result.start,
                result.end,
                result.entity_type,
                result.score,
            )
        )
    return formatted_results


def convert_to_recognizer_results(formatted_results):
    analyzer_results = []
    for result in formatted_results:
        # Handle both tuple format and RecognizerResult format
        if isinstance(result, tuple):
            text_value, start, end, entity_type, score = result
            recognizer_result = RecognizerResult(
                entity_type=entity_type, start=start, end=end, score=score
            )
        else:
            # If it's already a RecognizerResult, use it directly
            recognizer_result = result

        analyzer_results.append(recognizer_result)
    return analyzer_results


# TODO: overlook
def save_pattern_recognizer(custom_recognizer):
    """Save custom pattern recognizer to YAML file."""
    savefile = CUSTOM_RECOGNIZERS_FILE
    if custom_recognizer.name.startswith("NOT_"):
        savefile = NOT_RECOGNIZERS_FILE
    try:
        # Load existing recognizers if file exists
        existing_data = {"recognizers": []}
        if os.path.exists(savefile):
            with open(savefile, "r") as f:
                existing_data = yaml.safe_load(f) or {"recognizers": []}

        # check if already exists -> update
        checked_recognizer = check_existing_recognizer(custom_recognizer, existing_data)

        # Convert the new recognizer to YAML format with correct structure
        new_recognizer = {
            "name": checked_recognizer.name,
            "supported_language": checked_recognizer.supported_language,
            "supported_entity": checked_recognizer.supported_entities[0],
        }

        # Add patterns
        if hasattr(checked_recognizer, "patterns"):
            new_recognizer["patterns"] = [
                {
                    "name": pattern.name,
                    "regex": pattern.regex,
                    "score": pattern.score,
                }
                for pattern in checked_recognizer.patterns
            ]

        # Add context if it exists
        if hasattr(checked_recognizer, "context") and checked_recognizer.context:
            new_recognizer["context"] = checked_recognizer.context

        # Add deny_list only if it exists and is not empty
        if hasattr(checked_recognizer, "deny_list") and checked_recognizer.deny_list:
            new_recognizer["deny_list"] = checked_recognizer.deny_list

        # Remove existing recognizer if it exists
        existing_data["recognizers"] = [
            r
            for r in existing_data["recognizers"]
            if r["name"] != new_recognizer["name"]
        ]

        existing_data["recognizers"].append(new_recognizer)

        # Save to YAML file with proper formatting
        with open(savefile, "w") as f:
            yaml.dump(
                existing_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print("Pattern recognizer saved to YAML file.")

    except Exception as e:
        print(f"Error saving pattern recognizer to YAML: {e}")


def save_deny_list_recognizer(custom_recognizer):
    """Save custom deny list recognizer to YAML file."""
    savefile = CUSTOM_RECOGNIZERS_FILE
    if custom_recognizer.name.startswith("NOT_"):
        savefile = NOT_RECOGNIZERS_FILE
    try:
        # Load existing recognizers if file exists
        existing_data = {"recognizers": []}
        if os.path.exists(savefile):
            with open(savefile, "r") as f:
                existing_data = yaml.safe_load(f) or {"recognizers": []}

        # check if already exists -> update
        checked_recognizer = check_existing_recognizer(custom_recognizer, existing_data)

        # Convert the new recognizer to YAML format with correct structure
        new_recognizer = {
            "name": checked_recognizer.name,
            "supported_language": checked_recognizer.supported_language,
            "supported_entity": checked_recognizer.supported_entities[0],
        }

        # Add deny_list only if it exists and is not empty
        if hasattr(checked_recognizer, "deny_list") and checked_recognizer.deny_list:
            new_recognizer["deny_list"] = checked_recognizer.deny_list

        # Add context if it exists
        if hasattr(checked_recognizer, "context") and checked_recognizer.context:
            new_recognizer["context"] = checked_recognizer.context

        # Remove existing recognizer if it exists
        existing_data["recognizers"] = [
            r
            for r in existing_data["recognizers"]
            if r["name"] != new_recognizer["name"]
        ]

        # Add the updated recognizer
        existing_data["recognizers"].append(new_recognizer)

        # Save to YAML file with proper formatting
        with open(savefile, "w") as f:
            yaml.dump(
                existing_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print("Deny list recognizer saved to YAML file.")

    except Exception as e:
        print(f"Error saving deny list recognizer to YAML: {e}")


def check_existing_recognizer(custom_recognizer, existing_data):
    recognizer_name = custom_recognizer.name
    language = custom_recognizer.supported_language

    # Strip language suffix if present to get base name
    base_name = recognizer_name
    for lang in ["_de", "_en", "_fr", "_es"]:
        if base_name.endswith(lang):
            base_name = base_name[: -len(lang)]
            break

    # Look for existing recognizer with same base name and language
    existing_recognizer = next(
        (
            r
            for r in existing_data["recognizers"]
            if (
                r["name"] == recognizer_name  # Exact match
                or (
                    r["name"].startswith(base_name)  # Base name match
                    and r["supported_language"] == language
                )
            )  # Language match
        ),
        None,
    )

    if existing_recognizer:
        print(f"Recognizer {recognizer_name} already exists, updating...")

        # If existing recognizer doesn't have language suffix but should, update the name
        if not existing_recognizer["name"].endswith(f"_{language}"):
            recognizer_name = f"{base_name}_{language}"

        recognizer_kwargs = {
            "name": recognizer_name,
            "supported_entity": existing_recognizer["supported_entity"],
            "supported_language": language,
        }

        # Merge context lists
        if custom_recognizer.context or existing_recognizer.get("context", []):
            existing_context = existing_recognizer.get("context", [])
            merged_context = list(set(existing_context + custom_recognizer.context))
            recognizer_kwargs["context"] = merged_context

        # Merge deny lists
        if custom_recognizer.deny_list or existing_recognizer.get("deny_list", []):
            existing_deny_list = existing_recognizer.get("deny_list", [])
            merged_deny_list = list(
                set(existing_deny_list + custom_recognizer.deny_list)
            )
            recognizer_kwargs["deny_list"] = merged_deny_list

        # Merge patterns
        if hasattr(custom_recognizer, "patterns") or existing_recognizer.get(
            "patterns", []
        ):
            existing_patterns = existing_recognizer.get("patterns", [])
            new_patterns = getattr(custom_recognizer, "patterns", [])

            # Convert patterns to comparable format
            existing_pattern_tuples = {
                (p["regex"], p["score"]) for p in existing_patterns
            }
            new_pattern_tuples = {(p.regex, p.score) for p in new_patterns}

            # Merge patterns
            merged_pattern_tuples = existing_pattern_tuples.union(new_pattern_tuples)
            merged_patterns = [
                Pattern(name=f"pattern_{i}", regex=regex, score=score)
                for i, (regex, score) in enumerate(merged_pattern_tuples)
            ]
            recognizer_kwargs["patterns"] = merged_patterns

        # Remove existing recognizer from registry
        analyzer.registry.remove_recognizer(existing_recognizer["name"])
        return PatternRecognizer(**recognizer_kwargs)
    else:
        # For new recognizers, ensure they have language suffix
        if not recognizer_name.endswith(f"_{language}"):
            custom_recognizer.name = f"{recognizer_name}_{language}"
        return custom_recognizer


def load_custom_recognizers():
    """Load custom recognizers from all YAML files."""
    global custom_recognizers, registry
    try:
        custom_recognizers = []

        # Process each YAML file
        for yaml_file in RECOGNIZER_FILES:
            if os.path.exists(yaml_file):
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "recognizers" in data:
                        # Merge recognizers from this file
                        custom_recognizers.extend(data["recognizers"])

        # Clear existing custom recognizers from registry
        existing_names = [r.name for r in registry.recognizers]
        for name in existing_names:
            if name.endswith("recognizer") or any(
                name.endswith(f"recognizer_{lang}") for lang in ["de", "fr", "es", "en"]
            ):
                registry.remove_recognizer(name)

        # Add custom recognizers from all YAML files
        for yaml_file in RECOGNIZER_FILES:
            if os.path.exists(yaml_file):
                registry.add_recognizers_from_yaml(yaml_file)

        print(f"Loaded {len(custom_recognizers)} custom recognizers")

    except Exception as e:
        print(f"Error loading custom recognizers: {e}")


def remove_recognizer_from_savefile(name: str):
    """Remove a recognizer from all YAML files by name."""
    try:
        removed = False
        for yaml_file in RECOGNIZER_FILES:
            if os.path.exists(yaml_file):
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f) or {"recognizers": []}

                # Check if recognizer exists in this file
                original_count = len(data["recognizers"])
                data["recognizers"] = [
                    r for r in data["recognizers"] if r["name"] != name
                ]

                if len(data["recognizers"]) < original_count:
                    # Save the updated data back to the file
                    with open(yaml_file, "w") as f:
                        yaml.dump(
                            data,
                            f,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True,
                        )
                    print(f"Removed recognizer {name} from {yaml_file}")
                    removed = True

        return removed
    except Exception as e:
        print(f"Error removing recognizer from YAML: {e}")
        return False


# NEEDED?
def get_all_pattern_recognizers():
    """Get all pattern recognizers from registry."""

    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    # TODO: scan for patterns in custom recognizers
    # recognizers = analyzer.registry.recognizers
    # recognizers_list = []
    # for recognizer in recognizers:
    #     patterns = []
    #     if hasattr(recognizer, "patterns"):
    #         for pattern in recognizer.patterns:
    #             patterns.append(
    #                 {
    #                     "name": pattern.name,
    #                     "regex": pattern.regex,
    #                     "score": pattern.score,
    #                 }
    #             )
    #     recognizer_dict = {
    #         "name": recognizer.name,
    #         "supported_entity": (
    #             recognizer.supported_entities[0]
    #             if recognizer.supported_entities
    #             else None
    #         ),
    #         "patterns": patterns,
    #     }
    #     recognizers_list.append(recognizer_dict)

    # return recognizers_list


def get_custom_recognizers_from_registry():
    """Get only custom pattern recognizers from registry."""
    global custom_recognizers

    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    # Get custom recognizers from registry and ensure uniqueness by name
    seen_names = set()
    custom_recognizer_details = []

    for recognizer in analyzer.registry.recognizers:
        if (
            recognizer.name.endswith("recognizer")
            or any(
                recognizer.name.endswith(f"recognizer_{lang}")
                for lang in ["de", "fr", "es", "en"]
            )
            and recognizer.name not in seen_names
        ):
            seen_names.add(recognizer.name)
            custom_recognizer_details.append(
                {
                    "name": recognizer.name,
                    "supported_entity": (
                        recognizer.supported_entities[0]
                        if recognizer.supported_entities
                        else None
                    ),
                    "supported_language": recognizer.supported_language,
                    "patterns": (
                        [
                            {"name": p.name, "regex": p.regex, "score": p.score}
                            for p in recognizer.patterns
                        ]
                        if hasattr(recognizer, "patterns")
                        else None
                    ),
                    "context": (
                        recognizer.context if hasattr(recognizer, "context") else None
                    ),
                    "deny_list": (
                        recognizer.deny_list
                        if hasattr(recognizer, "deny_list")
                        else None
                    ),
                }
            )

    if custom_recognizer_details:
        print(f"Loaded {len(custom_recognizer_details)} unique custom recognizers: \n")
        print(json.dumps(custom_recognizer_details, indent=2, ensure_ascii=False))
    else:
        print("No custom recognizers found")

    return custom_recognizer_details


def get_custom_recognizers():
    """Get only custom pattern recognizers from registry."""
    global custom_recognizers

    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    print(f"GET {len(custom_recognizers)} custom recognizers: \n")
    print(custom_recognizers)

    return custom_recognizers


def get_all_recognizers():
    """Get all recognizers and their supported entities."""
    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    recognizers_info = []
    for recognizer in analyzer.registry.recognizers:
        # Handle different supported_entities types
        if not hasattr(recognizer, "supported_entities"):
            continue

        entities = recognizer.supported_entities
        if isinstance(entities, str):
            entities = [entities]
        elif not isinstance(entities, (list, tuple)):
            continue

        # Create entry for each entity
        for entity in entities:
            if entity:
                recognizers_info.append(
                    {"recognizer_name": recognizer.name, "supported_entity": entity}
                )

    return recognizers_info


def get_all_supported_entities():
    """Get all supported entities from recognizers."""
    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    entities = set()
    for recognizer in analyzer.registry.recognizers:
        if hasattr(recognizer, "supported_entities"):
            if not any(
                entity.startswith("NOT_") for entity in recognizer.supported_entities
            ):
                entities.update(recognizer.supported_entities)

    return list(entities)


def get_all_not_entities():
    """Get all supported entities from recognizers."""
    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    entities = set()
    for recognizer in analyzer.registry.recognizers:
        if hasattr(recognizer, "supported_entities"):
            if any(
                entity.startswith("NOT_") for entity in recognizer.supported_entities
            ):
                entities.update(recognizer.supported_entities)

    return list(entities)


def print_all_recognizers():
    if "analyzer" not in globals():
        create_analyzer_engine()
    else:
        load_custom_recognizers()

    print("All recognizers:")
    for r in analyzer.registry.recognizers:
        print(f"Name: {r.name}, Type: {type(r)}")


def print_colored_pii(string):
    colored_string = re.sub(
        r"(<[^>]*>)", lambda m: "\033[31m" + m.group(1) + "\033[0m", string
    )
    print(colored_string)
