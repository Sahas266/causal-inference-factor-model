import os # built-in Python module that allows code to interact with operating system
import time # pauses execution for a number of seconds when supabse or Internet fails for a second & executes again (eg: time.sleep(wait))
from typing import Any, Callable, Dict, List, Optional #  help understand what kind of data  expected.

from supabase import create_client # creates a Supabase client object ( create_client(url, key) )


Row = Dict[str, Any] # keys are strings (column names) & values can be anything (numbers, strings, dates, etc.) - its not the whole table just a single row (line) in table (spreadhseeet)
Rows = List[Row] # many rows for the SAME table


class SupabaseClient:
    """
    SupabaseClient is a small wrapper around supabase-py (the official Supabase Python library) # Our code never calls supabase-py directly, it allways goes through this class

    Responsibilities:
    - creates the connection using create_client & stores it in self.client & every method uses the same connection
    - Provide batch insert and batch upsert helpers (caller doesn't worry about batch sie)
    - if Supabase/network fails temporarily, the class waits and retries & callers don’t need retry logic

    It does NOT:
    - Fetch data from exchanges/APIs (only deals with supabase - fetching data belongs to other scripts)
    - Transform raw data into normalized form
    - Schedule jobs
    """

    def __init__(
        self,
        url: Optional[str] = None,
        service_role_key: Optional[str] = None,
        default_schema: str = "raw",
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        batch_size: int = 500,
    ) -> None: #this is the function signature
        """
        Create a SupabaseClient.

        Args:
            - url: Supabase project URL. (Optional[str] means: 	either a string	or None - can pass manually or from environment var.). If None, read SUPABASE_URL env var.
            - service_role_key: Supabase service role key (same ideas a url). If None, read SUPABASE_SERVICE_ROLE_KEY env var.
            - default_schema: Which  Postgres schema (folder) should I write to if none specified - (we assume raw).
            - max_retries: Number of attempts for transient failures, when Subase fails temporarily.
            - backoff_seconds: Base wait time for retries.
            -batch_size: Number of rows per API request (if ask for many rows, it may fail as API have limits).

        Raises:
            ValueError: If url or service_role_key is missing.
        """
        self.url = url or os.getenv("SUPABASE_URL") # use passed url (eg. client = SupabaseClient(url="https://your-project.supabase.co",service_role_key="your-secret-key") or read from system (eg. os.getenv("SUPABASE_URL")) 
        self.service_role_key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.url or not self.service_role_key:
            raise ValueError(
                "Missing credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )
        #Create Supabase client
        self.client = create_client(self.url, self.service_role_key)

        #Store config on the object
        self.default_schema = default_schema
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.batch_size = batch_size

    def insert_rows(
        self,
        table: str,
        rows: Rows,
        schema: Optional[str] = None,
    ) -> None: # never pass self manually, python does it for you!! 
        """
        Insert rows into the given table.

        Important:
            If you run it twice with the same rows, you may create duplicates (unless the table prevents it). 
            

        Args:
            table: Table name (without schema), e.g. "ohlcv_daily".
            rows: List of dict rows to insert.
            schema: Schema name, e.g. "raw". If None, uses default_schema (self.default_schema) which was defined in __init__

        Returns:
            None   Its purpose is side effects (writing to the database). If a error - exception will be raised.

        eg. client.upsert_rows(table="ohlcv_daily", rows=rows, schema="raw") -  Means put these rows into the table ohlcv_dail which lives inside the schema raw.
        """
        if not rows:
            return

        schema_to_use = schema or self.default_schema

        for chunk in self._chunk(rows, self.batch_size):
            self._retry(
                fn=lambda: self.client.schema(schema_to_use)
                .table(table)
                .insert(chunk)
                .execute()
            )

    def upsert_rows(
        self,
        table: str,
        rows: Rows,
        on_conflict: str,
        schema: Optional[str] = None,
    ) -> None:
        """
        Upsert rows into the given table. (Write many rows into a table in a way that is safe to run repeatedly)

        Upsert = insert if row doesn't exist, otherwise update existing row.
        If your 'on_conflict' matches a UNIQUE constraint / primary key, then
        upsert becomes idempotent (safe to run multiple times).

        Example for raw.ohlcv_daily:
            on_conflict="asset_id,day,source"
            Tells Supabase/Postgres: If a row already exists with the same asset_id AND day AND source, treat it as the same row.

            eg.
            -BTC, 2025-01-01, binance → one unique row	
            -BTC, 2025-01-01, coinbase → different row
            Hence multiple sources can coexist , but duplicates from the same source are prevented. What we want in raw

        Args:
            table: Table name (without schema), e.g. "ohlcv_daily".
            rows: List of dict rows to upsert.
            on_conflict: Comma-separated column names that define uniqueness.
            schema: Schema name. If None, uses default_schema.

        Returns
            None. Hence either succeeds silently or raises an exception
        """
        # if empty, nothing happens
        if not rows:
            return

        schema_to_use = schema or self.default_schema

        # batching the data ( APIs have limits, Large payloads fail more often, & Smaller batches are safer and faster to retry)
        for chunk in self._chunk(rows, self.batch_size):
            self._retry(
                fn=lambda: self.client.schema(schema_to_use)
                .table(table)
                .upsert(chunk, on_conflict=on_conflict)
                .execute()
            )

    def _retry(self, fn: Callable[[], Any]) -> None:
        """
        Execute fn() with retries and exponential backoff & only give up after max_retries

        Args:
            fn: A zero-argument function that performs the Supabase request.

        Raises:
            Exception: Re-raises the last exception if all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                fn()
                return
            except Exception as e:
                last_error = e
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                time.sleep(wait)

        # If we get here, all attempts failed
        raise last_error  # type: ignore

    @staticmethod # doesn't depend on object (self) - eg.  doesn't use self.client & hence _chunk has no self parameter
    def _chunk(rows: Rows, size: int) -> List[Rows]: # 
        """
        Take a big list of rows and split it into smaller lists (batches), cause APIs have limits & large payloads fail more often.o

        Args:
            rows: List of row dicts.
            size: Max rows per chunk.

        Returns:
            A list where each element is itself a Rows (a list of row dicts)
        """
        return [rows[i : i + size] for i in range(0, len(rows), size)]
















