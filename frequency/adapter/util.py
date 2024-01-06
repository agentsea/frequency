import re


def parse_gcs_uri(gcs_uri):
    """Parse a GCS URI into bucket and object key."""
    match = re.match(r"gs://([^/]+)/(.+)", gcs_uri)
    if not match:
        raise ValueError("Invalid GCS URI")
    return match.groups()
