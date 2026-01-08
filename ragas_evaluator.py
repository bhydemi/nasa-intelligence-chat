import os
from typing import Dict, List, Optional

# RAGAS imports with error handling
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from ragas import SingleTurnSample
    from ragas.metrics import (
        ResponseRelevancy,
        Faithfulness,
        BleuScore,
        RougeScore
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    IMPORT_ERROR = str(e)


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                             openai_api_key: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics

    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context strings
        openai_api_key: Optional OpenAI API key (uses env var if not provided)

    Returns:
        Dictionary with metric names and scores
    """
    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS not available: {IMPORT_ERROR}"}

    # Get API key from environment if not provided
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not provided"}

    try:
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0
            )
        )

        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
        )

        # Define an instance for each metric to evaluate
        response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness = Faithfulness(llm=evaluator_llm)
        bleu_score = BleuScore()
        rouge_score = RougeScore()

        # Prepare the sample for evaluation
        # Join contexts into a list if not already
        if isinstance(contexts, str):
            contexts = [contexts]

        # Create the evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Dictionary to store results
        results = {}

        # Evaluate each metric individually for better error handling
        metrics_to_evaluate = [
            ("response_relevancy", response_relevancy),
            ("faithfulness", faithfulness),
        ]

        for metric_name, metric in metrics_to_evaluate:
            try:
                score = metric.single_turn_score(sample)
                results[metric_name] = float(score) if score is not None else 0.0
            except Exception as e:
                results[metric_name] = 0.0
                results[f"{metric_name}_error"] = str(e)[:50]

        # BLEU and ROUGE need reference text - use context as reference
        if contexts:
            try:
                # For BLEU/ROUGE, create sample with reference
                reference_text = " ".join(contexts)[:1000]  # Limit reference length
                sample_with_ref = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    reference=reference_text
                )

                # BLEU score
                try:
                    bleu = bleu_score.single_turn_score(sample_with_ref)
                    results["bleu_score"] = float(bleu) if bleu is not None else 0.0
                except Exception:
                    results["bleu_score"] = 0.0

                # ROUGE score
                try:
                    rouge = rouge_score.single_turn_score(sample_with_ref)
                    results["rouge_score"] = float(rouge) if rouge is not None else 0.0
                except Exception:
                    results["rouge_score"] = 0.0

            except Exception as e:
                results["bleu_score"] = 0.0
                results["rouge_score"] = 0.0

        return results

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def batch_evaluate(test_data: List[Dict], openai_api_key: Optional[str] = None) -> Dict[str, any]:
    """
    Evaluate a batch of test questions

    Args:
        test_data: List of dictionaries with 'question', 'answer', 'contexts' keys
        openai_api_key: Optional OpenAI API key

    Returns:
        Dictionary with individual results and aggregated metrics
    """
    if not test_data:
        return {"error": "No test data provided"}

    results = {
        "individual_results": [],
        "aggregate_metrics": {}
    }

    # Collect scores for aggregation
    metric_scores = {
        "response_relevancy": [],
        "faithfulness": [],
        "bleu_score": [],
        "rouge_score": []
    }

    for i, item in enumerate(test_data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        contexts = item.get("contexts", [])

        if not question or not answer:
            results["individual_results"].append({
                "index": i,
                "error": "Missing question or answer"
            })
            continue

        # Evaluate this item
        scores = evaluate_response_quality(question, answer, contexts, openai_api_key)

        # Store individual result
        result = {
            "index": i,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "scores": scores
        }
        results["individual_results"].append(result)

        # Collect scores for aggregation
        for metric in metric_scores.keys():
            if metric in scores and isinstance(scores[metric], (int, float)):
                metric_scores[metric].append(scores[metric])

    # Calculate aggregates
    for metric, scores in metric_scores.items():
        if scores:
            results["aggregate_metrics"][metric] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }

    return results


def load_test_questions(filepath: str) -> List[Dict]:
    """
    Load test questions from a file

    Args:
        filepath: Path to test questions file (JSON or TXT)

    Returns:
        List of test question dictionaries
    """
    import json
    from pathlib import Path

    path = Path(filepath)

    if not path.exists():
        return []

    try:
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix == '.txt':
            # Parse text file - one question per line
            questions = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        questions.append({"question": line})
            return questions
        else:
            return []
    except Exception as e:
        print(f"Error loading test questions: {e}")
        return []
