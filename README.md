# Adaptive Bayesian User Modelling & Smart Advertisement System

## AI111 Course Project

**Group 16**

### Team Members

* Anakhveer Singh
* Devesh Dalal
* Devi Charan Mishra
* Dhanansh Pachauri
* Rishi Garg

---

## Abstract

This project implements an adaptive Bayesian user modelling and advertisement system using normalized vectors with values between **0 and 1**. The system improves traditional recommendation systems by prioritizing recent user behavior, preserving important long-term information, and solving the cold start problem for new users.

The user profile is continuously updated through Bayesian-style probability updates, and the system recommends the most relevant product, movie, advertisement, or suggestion by comparing the updated user vector with item vectors.

---

## Problem Statement

Many recommendation systems face these limitations:

### 1. Equal Treatment of Old and New Data

Older preferences may no longer represent the user’s current interests.

### 2. Poor Memory Management

Systems often fail to distinguish permanent preferences from temporary interests.

### 3. Cold Start Problem

New users have little or no data, reducing recommendation quality.

---

## Proposed Solution

We propose an intelligent advertising engine based on adaptive Bayesian updating with two separate memory layers.

### Long-Term Context

Stores stable information such as:

* Goals
* Academic interests
* Career preferences
* Persistent habits

### Short-Term Context

Stores changing preferences such as:

* Recent searches
* Current entertainment interests
* Session-based activity
* Temporary choices

---

## Core Mathematical Model

The user is represented by a normalized vector:

```text
(a, b, c, d, ...)
```

Where each value lies between:

```text
0 ≤ x ≤ 1
```

After new observations, the vector is updated to:

```text
(a₀, b₀, c₀, d₀, ...)
```

using adaptive Bayesian updating.

---

## Recommendation Logic

Each item in the inventory has its own feature vector.

Example:

```text
Movie_X = (0.10, 0.16, 0.90, 0.12)
```

The system compares vectors using:

```text
Cosine Similarity
```

or distance minimization:

```text
Σ(ai - a₀i)²
```

The item with highest relevance is recommended.

---

## Key Features

* Adaptive Bayesian Updating
* Recent Data Weighting
* Dual Memory Architecture
* Cold Start Initialization
* Cosine Similarity Recommendation Engine
* Multi-Domain User Modelling
* Dynamic Preference Learning

---

## Technologies Used

* Python
* NumPy
* Google Generative AI API
* Git & GitHub
* Probability Theory
* Bayesian Inference
* Linear Algebra

---

## Project Structure

```bash
AI111-Probabilistic-User-Modelling/
│── README.md
│── LICENSE
│── requirements.txt
│── src/
│   ├── main.py
│   ├── mainsimulation.py
│   ├── coreaiengine.py
│   ├── mathshelperfunctions.py
│   └── setupllm.py
```

---

## File Descriptions

### main.py

Main entry point of the project. Runs the recommendation engine and user interaction flow.

### mainsimulation.py

Handles simulation logic for testing user behavior and recommendations.

### coreaiengine.py

Contains the core adaptive Bayesian model, memory system, profiling, and recommendation logic.

### mathshelperfunctions.py

Contains helper mathematical functions such as vector length, cosine similarity, and calculations.

### setupllm.py

Configures and connects the Google Generative AI API used for NLP-based intent parsing.

### requirements.txt

Contains required Python libraries to run the project.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/AI111-Probabilistic-User-Modelling.git
cd AI111-Probabilistic-User-Modelling
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Open `src/setupllm.py` or `src/main.py` and replace:

```python
API_KEY = "ENTER_API_KEY_HERE"
```

with your actual API key.

### 4. Run Project

```bash
python src/main.py
```

---

## How It Works

1. Create an initial blank or statistical user vector.
2. Accept user text input.
3. Use NLP to convert text into weighted preference categories.
4. Update short-term and long-term memory vectors.
5. Compare updated profile with item vectors.
6. Recommend the best matching output.
7. Continue learning from future inputs.

---

## Example Use Case

Initial User Vector:

```text
(0.20, 0.10, 0.40, 0.30)
```

After new user activity:

```text
(0.15, 0.08, 0.60, 0.17)
```

Possible Outputs:

```text
Action_Movie = (0.10, 0.05, 0.90, 0.05)
Romance_Movie = (0.05, 0.95, 0.02, 0.03)
```

Recommended Output: **Action_Movie**

---

## Future Scope

* Reinforcement learning integration
* Better automatic question generation
* Web dashboard for visualization
* Real-time analytics
* Large-scale deployment
* Use in education, healthcare, e-commerce, and entertainment

---

## GitHub Description

```text
This project implements an adaptive Bayesian user modelling and smart recommendation system using 0–1 normalized vectors, recent-data prioritization, dual memory context, cold start handling, and vector similarity optimization.
```

---

## Conclusion

This project demonstrates how Bayesian reasoning and vector-based modelling can create smarter, more adaptive, and more transparent recommendation systems that understand changing user preferences over time.

---

## License

MIT License

Copyright (c) 2026 devesh2007777

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
