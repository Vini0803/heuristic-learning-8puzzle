# Heuristic Learning for the 8-Puzzle

This project investigates whether a learned combination of classical heuristics can approximate the real cost of the 8-Puzzle problem better than each heuristic individually.

## 📌 Objective

The main goal of this project is to test the following hypothesis:

> Can a learned combination of the Manhattan and Hamming heuristics approximate the real cost more accurately than each heuristic alone?

To answer this question, we apply supervised learning (Linear Regression) to combine both heuristics and evaluate the results.

---

## 🧠 Methodology

### 1️⃣ Real Cost Computation (Ground Truth)

- The complete state space of the 8-Puzzle (181,440 reachable states) was generated.
- A **Breadth-First Search (BFS)** algorithm was executed starting from the goal state.
- The optimal cost for every reachable state was stored in a dictionary (`dist`).

This `dist` dictionary represents the **true optimal cost** and was not modified at any point.

---

### 2️⃣ Classical Heuristics

For each state, two admissible heuristics were computed:

- **Manhattan Distance**
- **Hamming Distance**

These values were used as input features for the supervised learning model.

---

### 3️⃣ Supervised Learning Model

A **Linear Regression** model was trained using:

- **Input (X):** `[Manhattan, Hamming]`
- **Target (y):** Real cost obtained from BFS (`dist`)

The learned heuristic has the form:
h(s) = a + b * Manhattan + c * Hamming


The coefficients were automatically learned by minimizing the mean squared error.

---

## 📊 Results

The performance was evaluated using Mean Absolute Error (MAE):

- Manhattan MAE ≈ 7.97  
- Hamming MAE ≈ 14.86  
- Learned Model MAE ≈ 2.18  

The learned heuristic significantly reduced the average prediction error.

---

## ⚖️ Admissibility Analysis

Although the learned model reduced the average error, it **lost admissibility**.

- Approximately 48% of states were overestimated.
- Therefore, the learned heuristic cannot guarantee optimality if used directly in algorithms such as A*.

This demonstrates an important trade-off:

- Minimizing statistical error  
vs  
- Preserving theoretical guarantees (admissibility)

---

## ✅ Conclusion

The hypothesis was confirmed in terms of numerical accuracy:

The learned combination of Manhattan and Hamming heuristics approximates the real cost significantly better than each heuristic individually.

However, the learned heuristic is not admissible, highlighting the difference between statistical optimization and theoretical guarantees in search algorithms.

---

## 🛠 Technologies Used

- Python
- NumPy
- Scikit-learn
- BFS (graph search)

---

## 📚 Key Concepts

- Heuristic Functions
- 8-Puzzle
- Breadth-First Search (BFS)
- Supervised Learning
- Linear Regression
- Admissibility in A*

---

## 👤 Author

Vinícius Milomem Santos
