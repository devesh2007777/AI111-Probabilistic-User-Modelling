# Adaptive Bayesian User Modelling & Smart Recommendation System

## AI111 Course Project

**Group 16**

### Team Members

* Anakhveer Singh
* Devesh Dalal
* Devi Charan Mishra
* Dhanansh Pachauri
* Rishi Garg

---

# Abstract

This project focuses on building a more efficient Bayesian update model for intelligent recommendation and decision systems. Current AI systems often treat old and new user data similarly, struggle with memory management, and face the cold start problem for new users.

Our proposed system solves these challenges by:

1. Giving higher priority to recent user behavior than outdated behavior.
2. Separating memory into **long-term context** and **short-term context**.
3. Handling new users using statistical initialization.

The model continuously updates a user preference vector and recommends outputs whose feature vectors are closest to the updated user state.

---

# Problem Statement

Modern AI recommendation systems face three common problems:

### 1. Old Data vs Recent Data

A user’s interests can change over time. Recent preferences should matter more than old ones.

### 2. What to Remember and What to Forget

Some user information is permanent (college, branch, career goals), while some is temporary (current mood, short-term interests).

### 3. Cold Start Problem

For new users, the system has no past data and cannot make good recommendations initially.

---

# Proposed Solution

We propose an **Adaptive Bayesian User Model** with two memory layers:

## Long-Term Context Vector

Stores stable information such as:

* Future goals
* College
* Branch
* Career interests
* Persistent preferences

## Short-Term Context Vector

Stores temporary or changing information such as:

* Recent movie genre interest
* Current shopping preference
* Trending choices
* Session-based behavior

---

# Core Idea

The user is represented by an initial preference vector:

```text id="r31326"
(a, b, c, d)
```

Example:

```text id="i74742"
(Horror, Romance, Action, Comedy)
```

Using Bayesian updating, the system transforms it into:

```text id="s23995"
(a₀, b₀, c₀, d₀)
```

where updated values reflect new observations and recent behavior.

---

# Recommendation Logic

Each output item also has its own feature vector.

Example:

```text id="p23829"
Singham = (0.1, 16, 90, 12.9)
```

The system recommends the item whose vector is closest to the updated user vector by minimizing:

```text id="x35914"
Σ(ai - a₀i)²
```

This is the squared distance measure used to find the best match.

---

# Objectives

* Build an efficient Bayesian update model.
* Prioritize recent data over older data.
* Separate long-term and short-term memory.
* Solve the cold start problem for new users.
* Recommend outputs using vector similarity.
* Improve personalization and accuracy.

---

# Features

* **Bayesian Preference Updating**
* **Weighted Recent Data Handling**
* **Dual Memory Context System**
* **Cold Start Initialization**
* **Vector-Based Recommendation Engine**
* **Scalable Design for Future AI Systems**

---

# Technologies Used

* Python
* NumPy
* Git & GitHub
* Bayesian Inference
* Linear Algebra
* Probability Theory

---

# How It Works

## Step 1: Initialize User Vector

Create a starting preference vector based on known data or statistics.

## Step 2: Observe New Inputs

Collect recent user actions, choices, or answers.

## Step 3: Bayesian Update

Update each component of the vector efficiently.

## Step 4: Memory Management

Store stable data in long-term memory and temporary data in short-term memory.

## Step 5: Compare with Output Vectors

Calculate distance between user vector and item vectors.

## Step 6: Recommend Best Match

Return the item with minimum squared error.

---

# Example Use Case

Initial User Vector:

```text id="p52797"
(20, 10, 40, 30)
```

After recent activity favoring Action movies:

```text id="m63287"
(15, 8, 60, 17)
```

Movie Vectors:

```text id="j51926"
Singham = (10, 5, 90, 5)
Titanic = (5, 95, 2, 3)
```

The system recommends **Singham** because it is closer to the updated user vector.

---

# Future Scope

* Automatic question generation to learn preferences faster
* Reinforcement learning integration
* Real-time user adaptation
* Web dashboard for visualization
* Large-scale recommendation deployment
* Use in education, entertainment, healthcare, and e-commerce

---

# Project Structure

```bash id="s65328"
AI111-Project/
│── README.md
│── src/
│   ├── main.py
│   ├── bayesian_update.py
│   ├── recommender.py
│── docs/
│── report.pdf
│── presentation/
│── requirements.txt
```

---

# GitHub Contribution

Repository used for:

* Version control
* Team collaboration
* Code management
* Documentation
* Project tracking

---

# Conclusion

This project presents a smarter Bayesian recommendation system that updates user preferences dynamically, remembers important long-term information, adapts to short-term changes, and gives better recommendations through vector optimization.

---

# License

This project is developed for academic purposes under the AI111 course.
