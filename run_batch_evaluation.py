#!/usr/bin/env python3
"""
Batch Evaluation Runner for NASA RAG System

This script runs end-to-end evaluation using the RAG system and RAGAS metrics.
It loads test questions, generates answers using the RAG pipeline, and evaluates
the responses.
"""

import argparse
import json
import os
from typing import Dict, List

import rag_client
import llm_client
import ragas_evaluator


def run_batch_evaluation(
    openai_key: str,
    chroma_dir: str,
    collection_name: str,
    test_file: str,
    n_results: int = 3,
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Run batch evaluation on test questions

    Args:
        openai_key: OpenAI API key
        chroma_dir: ChromaDB directory
        collection_name: Collection name
        test_file: Path to test questions file
        n_results: Number of documents to retrieve
        model: LLM model to use

    Returns:
        Evaluation results dictionary
    """
    # Load test questions
    print(f"Loading test questions from {test_file}...")
    test_questions = ragas_evaluator.load_test_questions(test_file)

    if not test_questions:
        return {"error": f"No test questions found in {test_file}"}

    print(f"Loaded {len(test_questions)} test questions")

    # Initialize RAG system
    print(f"Initializing RAG system from {chroma_dir}/{collection_name}...")
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)

    if not success:
        return {"error": f"Failed to initialize RAG system: {error}"}

    print("RAG system initialized successfully")

    # Process each question
    results = {
        "total_questions": len(test_questions),
        "individual_results": [],
        "aggregate_metrics": {}
    }

    metric_scores = {
        "response_relevancy": [],
        "faithfulness": [],
        "bleu_score": [],
        "rouge_score": []
    }

    for i, item in enumerate(test_questions):
        question = item.get("question", "")
        category = item.get("category", "unknown")
        mission = item.get("mission", None)

        print(f"\n[{i+1}/{len(test_questions)}] Processing: {question[:50]}...")

        if not question:
            results["individual_results"].append({
                "index": i,
                "error": "Empty question"
            })
            continue

        try:
            # Retrieve documents
            docs_result = rag_client.retrieve_documents(
                collection,
                question,
                n_results,
                mission_filter=mission
            )

            # Format context
            context = ""
            contexts_list = []
            if docs_result and docs_result.get("documents"):
                context = rag_client.format_context(
                    docs_result["documents"][0],
                    docs_result["metadatas"][0]
                )
                contexts_list = docs_result["documents"][0]

            # Generate response
            response = llm_client.generate_response(
                openai_key,
                question,
                context,
                [],  # No conversation history for batch
                model
            )

            print(f"   Answer: {response[:100]}...")

            # Evaluate response
            scores = ragas_evaluator.evaluate_response_quality(
                question,
                response,
                contexts_list,
                openai_key
            )

            # Store result
            result = {
                "index": i,
                "question": question,
                "category": category,
                "mission": mission,
                "answer": response,
                "context_count": len(contexts_list),
                "scores": scores
            }
            results["individual_results"].append(result)

            # Collect scores for aggregation
            for metric in metric_scores.keys():
                if metric in scores and isinstance(scores[metric], (int, float)):
                    metric_scores[metric].append(scores[metric])

            # Print scores
            print(f"   Scores: ", end="")
            for metric, score in scores.items():
                if isinstance(score, (int, float)):
                    print(f"{metric}={score:.3f} ", end="")
            print()

        except Exception as e:
            results["individual_results"].append({
                "index": i,
                "question": question,
                "error": str(e)
            })
            print(f"   Error: {e}")

    # Calculate aggregates
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)

    for metric, scores in metric_scores.items():
        if scores:
            agg = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
            results["aggregate_metrics"][metric] = agg
            print(f"{metric}: mean={agg['mean']:.3f}, min={agg['min']:.3f}, max={agg['max']:.3f} (n={agg['count']})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run batch evaluation on NASA RAG system')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--test-file', default='test_questions.json', help='Test questions file (JSON or TXT)')
    parser.add_argument('--n-results', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file for results')

    args = parser.parse_args()

    # Set API key in environment
    os.environ["OPENAI_API_KEY"] = args.openai_key

    print("=" * 60)
    print("NASA RAG BATCH EVALUATION")
    print("=" * 60)

    # Run evaluation
    results = run_batch_evaluation(
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        test_file=args.test_file,
        n_results=args.n_results,
        model=args.model
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total questions: {results.get('total_questions', 0)}")
    print(f"Successful evaluations: {len([r for r in results.get('individual_results', []) if 'error' not in r])}")
    print(f"Failed evaluations: {len([r for r in results.get('individual_results', []) if 'error' in r])}")


if __name__ == "__main__":
    main()
