# Supabase Python Integration Guide

This document provides a complete guide for integrating **Supabase** with Python using the official `supabase-py` client. It covers installation, initialization, database operations, authentication, storage, Edge Functions, and best practices.

---

## 1. Overview

Supabase is an open-source backend-as-a-service built on Postgres. The Python client (`supabase-py`) allows you to:

* Interact with a Postgres database (CRUD, RPCs, filters)
* Manage authentication and users
* Upload and retrieve files from storage
* Subscribe to realtime events
* Invoke Edge Functions

The Python client is suitable for:

* Backend services
* Data pipelines
* Quant / analytics workloads
* Automation and bots

---

## 2. Prerequisites

* Python **3.9+**
* A Supabase project
* Supabase **Project URL**
* Supabase **API Key** (anon or service role)

> ⚠️ Never expose the **service role key** in client-facing environments.

---

## 3. Installation

Install via pip:

```bash
pip install supabase
```

Optional (recommended for production):

```bash
pip install python-dotenv
```

---

## 4. Environment Configuration

Store credentials securely using environment variables.

```bash
SUPABASE_URL=https://<project-id>.supabase.co
SUPABASE_KEY=<your-api-key>
```

Load them in Python:

```python
import os
from supabase import create_client, Client

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]

supabase: Client = create_client(url, key)
```

---

## 5. Database Operations (CRUD)

### 5.1 Select (Read)

```python
response = (
    supabase
    .table("assets")
    .select("*")
    .execute()
)

data = response.data
```

Select specific columns:

```python
supabase.table("assets").select("symbol, price").execute()
```

---

### 5.2 Insert (Create)

```python
supabase.table("assets").insert({
    "symbol": "BTC",
    "price": 43000
}).execute()
```

Bulk insert:

```python
supabase.table("assets").insert([
    {"symbol": "ETH", "price": 2300},
    {"symbol": "SOL", "price": 98}
]).execute()
```

---

### 5.3 Update

Updates **must** be combined with filters.

```python
supabase.table("assets") \
    .update({"price": 44000}) \
    .eq("symbol", "BTC") \
    .execute()
```

---

### 5.4 Delete

```python
supabase.table("assets") \
    .delete() \
    .eq("symbol", "SOL") \
    .execute()
```

---

## 6. Filtering & Querying

Common filters:

```python
.eq("column", value)
.neq("column", value)
.gt("column", value)
.gte("column", value)
.lt("column", value)
.lte("column", value)
.in_("column", [v1, v2])
.like("column", "%pattern%")
.ilike("column", "%pattern%")
```

Example:

```python
supabase.table("trades") \
    .select("*") \
    .gte("timestamp", "2025-01-01") \
    .lt("price", 50000) \
    .execute()
```

---

## 7. Pagination & Limits

Limit rows:

```python
supabase.table("trades").select("*").limit(100).execute()
```

Range-based pagination:

```python
supabase.table("trades") \
    .select("*") \
    .range(0, 99) \
    .execute()
```

---

## 8. Single-Row Queries

Return exactly one row:

```python
supabase.table("assets") \
    .select("*") \
    .eq("symbol", "BTC") \
    .single() \
    .execute()
```

Return zero or one row safely:

```python
supabase.table("assets") \
    .select("*") \
    .eq("symbol", "DOGE") \
    .maybe_single() \
    .execute()
```

---

## 9. Upserts

Insert or update on conflict:

```python
supabase.table("assets").upsert({
    "symbol": "BTC",
    "price": 45000
}).execute()
```

With explicit conflict target:

```python
supabase.table("assets").upsert(
    {"symbol": "ETH", "price": 2400},
    on_conflict="symbol"
).execute()
```

---

## 10. Calling Postgres Functions (RPC)

Create a function in Postgres:

```sql
create or replace function total_volume()
returns numeric as $$
  select sum(volume) from trades;
$$ language sql;
```

Call it from Python:

```python
supabase.rpc("total_volume").execute()
```

With parameters:

```python
supabase.rpc("get_trades_by_symbol", {"symbol": "BTC"}).execute()
```

---

## 11. Authentication

### 11.1 Sign Up

```python
supabase.auth.sign_up({
    "email": "user@example.com",
    "password": "secure-password"
})
```

---

### 11.2 Sign In

```python
supabase.auth.sign_in_with_password({
    "email": "user@example.com",
    "password": "secure-password"
})
```

---

### 11.3 Get Current User

```python
supabase.auth.get_user()
```

---

### 11.4 Sign Out

```python
supabase.auth.sign_out()
```

---

## 12. Storage (Files)

### Upload File

```python
with open("model.pkl", "rb") as f:
    supabase.storage \
        .from_("models") \
        .upload("model.pkl", f)
```

### Download File

```python
supabase.storage \
    .from_("models") \
    .download("model.pkl")
```

---

## 13. Edge Functions

Invoke an Edge Function:

```python
supabase.functions.invoke(
    "run-simulation",
    invoke_options={
        "body": {"scenario": "market-crash"}
    }
)
```

---

## 14. Realtime (Advanced)

Subscribe to database changes (long-lived process):

```python
channel = supabase.channel("realtime:trades")

channel.on(
    "postgres_changes",
    {
        "event": "*",
        "schema": "public",
        "table": "trades",
    },
    lambda payload: print(payload)
)

channel.subscribe()
```

---

## 15. Error Handling

Always inspect responses:

```python
response = supabase.table("assets").select("*").execute()

if response.error:
    raise Exception(response.error.message)
```

---

## 16. Security & Best Practices

* Use **RLS (Row Level Security)** aggressively
* Prefer **RPCs** for complex business logic
* Use **service role key** only on trusted servers
* Avoid unbounded `.select("*")` in production
* Log all mutations in critical systems (trading, auth)

---

## 17. Recommended Project Structure

```text
app/
├── db/
│   ├── client.py
│   └── queries.py
├── services/
│   ├── auth.py
│   ├── storage.py
│   └── analytics.py
├── config.py
└── main.py
```

---

## 18. Summary

The Supabase Python client provides a powerful, typed, and composable interface to Postgres-backed infrastructure. Used correctly, it can replace:

* ORMs
* Auth providers
* File storage services
* Lightweight APIs

For high-performance or institutional workloads, combine Supabase with:

* RPC-based logic
* Strict RLS
* Async workers
* Deterministic schemas

---

**You now have a production-ready Supabase Python integration.**
