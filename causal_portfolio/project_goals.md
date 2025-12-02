# **Structural Causal Model (SCM)** for portfolio optimization using **DoWhy** with optional EconML hooks

* Module structure
* Factor node definitions
* Shock (IV) construction
* SCM graph creation
* Identification + estimation templates

---

# **Project Structure**

```
causal_portfolio/
│
├── data/
│   ├── future db integration
│
├── scm/
│   ├── __init__.py
│   ├── graph.py
│   ├── shocks.py
│   ├── loaders.py
│   ├── model.py
│   └── estimation.py
│
└── main.py
```

---

# **Module responsibility**

### **scm/loaders.py — load returns, factors, shocks**
Once the db is set up we will replace these mock csv loaders with a supabase connector

### **scm/shocks.py — construct your IVs exogenous shock series**
This gives you **one instrument per factor** 

### **scm/graph.py — define the causal DAG**

### **scm/model.py — instantiate the DoWhy SCM**

### **scm/estimation.py — identify + estimate causal effects**

### **main.py — wire everything together**

---
