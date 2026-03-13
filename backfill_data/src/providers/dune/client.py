"""Dune Analytics API v1 client"""

import requests
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger('backfill_system.dune')


class DuneClient:
    """
    HTTP client for the Dune Analytics API v1.

    Base URL: ``https://api.dune.com/api/v1``
    Auth: ``X-Dune-Api-Key`` header.

    Supported operations
    --------------------
    get_query_results(query_id)
        Fetch the latest cached results for a pre-saved query.

    execute_query(query_id, params)
        Trigger a fresh execution of a saved query with optional parameters.

    get_execution_status(execution_id)
        Poll execution state (QUERY_STATE_COMPLETED, QUERY_STATE_EXECUTING, etc.).

    get_execution_results(execution_id)
        Fetch results from a specific execution.

    Note: Queries must be created in the Dune UI first. The API cannot
    run ad-hoc SQL — it only operates on saved query IDs.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.dune.com/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'X-Dune-Api-Key': self.api_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        })
        logger.info(f"Dune client initialised: {self.base_url}")

    # ------------------------------------------------------------------
    # Query results (cached / latest)
    # ------------------------------------------------------------------

    def get_query_results(
        self,
        query_id: int,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict:
        """
        GET /query/{query_id}/results

        Fetch the latest cached results for a saved query.

        Returns:
            {
                "execution_id": "...",
                "state": "QUERY_STATE_COMPLETED",
                "result": {
                    "rows": [...],
                    "metadata": {"column_names": [...], ...}
                }
            }
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
        return self._get(f"{self.base_url}/query/{query_id}/results", params)

    # ------------------------------------------------------------------
    # Query execution (trigger fresh run)
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query_id: int,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        POST /query/{query_id}/execute

        Trigger a fresh execution of a saved query.

        Args:
            query_id: The saved query ID from the Dune UI.
            params: Optional query_parameters to pass to the query.

        Returns:
            {"execution_id": "..."}
        """
        body: Dict[str, Any] = {}
        if params:
            body['query_parameters'] = params
        return self._post(f"{self.base_url}/query/{query_id}/execute", body)

    # ------------------------------------------------------------------
    # Execution status / results
    # ------------------------------------------------------------------

    def get_execution_status(self, execution_id: str) -> Dict:
        """
        GET /execution/{execution_id}/status

        Returns:
            {
                "execution_id": "...",
                "state": "QUERY_STATE_COMPLETED|QUERY_STATE_EXECUTING|...",
                ...
            }
        """
        return self._get(f"{self.base_url}/execution/{execution_id}/status")

    def get_execution_results(
        self,
        execution_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict:
        """
        GET /execution/{execution_id}/results

        Fetch results from a specific execution. Same response format
        as get_query_results.
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
        return self._get(f"{self.base_url}/execution/{execution_id}/results", params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        try:
            logger.debug(f"GET {url} params={params}")
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(
                f"HTTP {e.response.status_code} from {url}: "
                f"{e.response.text[:500] if e.response else 'N/A'}"
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Request error from {url}: {e}")
            raise

    def _post(self, url: str, body: Dict) -> Any:
        import json
        try:
            logger.debug(f"POST {url}")
            response = self.session.post(url, data=json.dumps(body), timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(
                f"HTTP {e.response.status_code} from {url}: "
                f"{e.response.text[:500] if e.response else 'N/A'}"
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Request error posting to {url}: {e}")
            raise

    def close(self):
        self.session.close()
        logger.debug("Dune client session closed")
