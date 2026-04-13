"""
Airtable Client Module.

Handles connection and schema operations for Airtable.
"""

import logging
import requests
from pyairtable import Api

from ..config import (
    AIRTABLE_API_KEY,
    AIRTABLE_BASE_ID,
    AIRTABLE_TABLE_ID,
    NEW_SCORE_FIELD
)


logger = logging.getLogger(__name__)


def get_airtable_table():
    """
    Initialize Airtable connection.

    Returns:
        Tuple of (Api, Table) objects
    """
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)
    return api, table


def ensure_field_exists(api) -> bool:
    """
    Ensure the Pronunciation Assessment Score field exists in the table.
    Creates it if it doesn't exist.

    Args:
        api: Airtable Api object

    Returns:
        True if field exists or was created, False on error
    """
    try:
        base = api.base(AIRTABLE_BASE_ID)
        schema = base.schema()

        # Find the Test Results table
        test_results_table = None
        for table in schema.tables:
            if table.id == AIRTABLE_TABLE_ID:
                test_results_table = table
                break

        if not test_results_table:
            logger.error("Test Results table not found")
            return False

        # Check if field exists
        field_exists = any(
            field.name == NEW_SCORE_FIELD
            for field in test_results_table.fields
        )

        if field_exists:
            logger.info(f"Field '{NEW_SCORE_FIELD}' already exists")
            return True

        # Create the field
        logger.info(f"Creating field '{NEW_SCORE_FIELD}'...")

        url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables/{AIRTABLE_TABLE_ID}/fields"
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": NEW_SCORE_FIELD,
            "type": "number",
            "options": {
                "precision": 1
            }
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info(f"Field '{NEW_SCORE_FIELD}' created successfully")
            return True
        else:
            logger.error(f"Failed to create field: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error checking/creating field: {e}")
        return False
