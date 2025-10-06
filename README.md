
# LLM-on-Premise

**Benchmarking LLMs across on-premise and cloud environments** — unified chat interface, optimized chunking pipeline, and controlled evaluation of DeepSeek (Ollama) vs. OpenAI (Azure).

---

## Purpose

This project investigates how **local deployment (Ollama + DeepSeek)** compares to **cloud-based inference (OpenAI on Azure)** in terms of **latency, reliability, and output quality**.
It was built to explore **real-world trade-offs between inference locality, API abstraction, and system control** — a key concern in scalable AI infrastructures.

---

## Core Components

* **Unified Chat UI**
  Interactive interface to query both local and remote models under identical conditions.
  Terminal logs provide detailed traces of request/response cycles and token behavior.

* **Chunking & Overlap Management**
  Custom segmentation algorithm to handle overlapping text chunks efficiently.

* **Dual Backend Integration**

  * **DeepSeek** through **Ollama** (on-premise inference)
  * **OpenAI** through **Azure API** (cloud inference)
    Both integrated under a common abstraction layer for seamless switching and testing.

---

## Evaluation & Benchmarking

A **dedicated benchmark** was conducted to measure accuracy, consistency, and relevance between the two systems.
The test used a **set of predefined prompts** and a **custom scoring protocol** to rank responses.

> The benchmark code and results are not included in this repository (used internally for reporting and analysis).

---

## Technical Highlights

* Python backend managing request routing, scoring, and segmentation.
* Modular design enabling model substitution or hybrid setups.
* Compatible with both **interactive** and **scripted** evaluation modes.
* Built with scalability and reproducibility in mind.

---

## Contributors

* **Alexadnre**
* **IIsalSaili**
* **tlemenC**
* **G4Bi-12**
* **nlhmnlhmnlhm**
* **LEMENNN**
