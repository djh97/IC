# Methodology Tracking

This document tracks what the implementation must preserve so the later paper can be written without reconstructing decisions from memory.

## Methods We Must Be Able To Report

- model name, provider, and endpoint configuration
- retrieval configuration
- source categories and authority
- chunking and indexing procedure
- personalization workflow
- traceability mechanism
- structured output schema
- evaluation setup and baselines

## Results We Must Be Able To Produce

- coverage outcomes
- traceability outcomes
- readability outcomes
- personalization outcomes
- structured-output validity
- ablation comparisons
- qualitative case examples

## Implementation Rule

Any stage that could affect the final paper should emit an artifact, log entry, or config snapshot.
