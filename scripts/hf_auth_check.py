"""Early HF token / dataset-namespace check for Vast.ai sessions.

Stdlib-only on purpose: this must run BEFORE step-1 dependency installation on a
fresh box, so it cannot import numpy/src/huggingface_hub. It verifies that
HF_TOKEN is valid (whoami-v2) and that HF_DATASET_REPO's namespace exactly
matches the token owner's username (case-sensitive) with write permission.

The token value is never printed or logged.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


class HfAuthError(ValueError):
    """Raised when HF auth preflight fails; message is user-facing."""


def fetch_whoami(token: str, endpoint: str = os.getenv("HF_ENDPOINT", "https://huggingface.co")) -> dict:
    """GET {endpoint}/api/whoami-v2 with a bearer token. Never echoes the token."""
    url = endpoint.rstrip("/") + "/api/whoami-v2"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise HfAuthError(
                "HF token invalid or expired (whoami-v2 returned 401). "
                f"Create a new token at {TOKEN_SETTINGS_URL}"
            ) from exc
        raise HfAuthError(f"HF whoami-v2 request failed with HTTP {exc.code}.") from exc
    except HfAuthError:
        raise
    except Exception as exc:
        raise HfAuthError(f"HF whoami-v2 request failed: {exc}") from exc


def _token_role_and_fine_grained(whoami: dict) -> tuple[str, dict]:
    try:
        access = whoami["auth"]["accessToken"]
        role = access["role"]
    except (KeyError, TypeError) as exc:
        raise HfAuthError(
            "HF whoami-v2 response is missing auth.accessToken.role; cannot verify token permissions."
        ) from exc
    fine_grained = access.get("fineGrained") or {}
    return str(role), fine_grained if isinstance(fine_grained, dict) else {}


def validate_hf_identity(whoami: dict, repo_id: str) -> str:
    """Validate repo namespace vs token owner + write permission. Returns OK message."""
    if not isinstance(repo_id, str):
        raise HfAuthError(f"HF_DATASET_REPO must be a string in 'namespace/name' form, got {repo_id!r}.")
    repo_id = repo_id.strip()
    parts = repo_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1] or any(c.isspace() for c in repo_id):
        raise HfAuthError(
            f"HF_DATASET_REPO '{repo_id}' must be exactly 'namespace/name' (e.g. \"Cipoklean/cloakbreak-data-v1\")."
        )
    namespace, name = parts

    role, fine_grained = _token_role_and_fine_grained(whoami)
    if role == "fineGrained":
        scoped = fine_grained.get("scoped") or []
        has_repo_write = any(
            isinstance(entry, dict)
            and isinstance(entry.get("entity"), dict)
            and entry["entity"].get("name") == namespace
            and "repo.write" in (entry.get("permissions") or [])
            for entry in scoped
        )
        if not has_repo_write:
            raise HfAuthError(
                f"HF token role is 'fineGrained' but has no 'repo.write' permission scoped to namespace "
                f"'{namespace}'. Session uploads need a write token (create one at {TOKEN_SETTINGS_URL})."
            )
    elif role != "write":
        raise HfAuthError(
            f"HF token role is '{role}'; Session uploads need a write token "
            f"(create one at {TOKEN_SETTINGS_URL})"
        )

    owner = whoami.get("name")
    if owner != namespace:
        raise HfAuthError(
            f"HF_DATASET_REPO namespace '{namespace}' does not match token owner '{owner}' (case-sensitive). "
            f"Fix: export HF_DATASET_REPO=\"{owner}/{name}\""
        )
    return f"HF auth OK: user={owner} role={role} repo={repo_id}"


def main() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_DATASET_REPO", "").strip()
    try:
        if not token:
            raise HfAuthError("HF_TOKEN is not set; export a Hugging Face token first.")
        if not repo_id:
            raise HfAuthError(
                "HF_DATASET_REPO is not set; export e.g. HF_DATASET_REPO=\"Cipoklean/cloakbreak-data-v1\"."
            )
        whoami = fetch_whoami(token)
        message = validate_hf_identity(whoami, repo_id)
    except HfAuthError as exc:
        sys.exit(f"HF AUTH PREFLIGHT FAIL: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        sys.exit(f"HF AUTH PREFLIGHT FAIL: unexpected error: {exc}")
    print(message)


if __name__ == "__main__":
    main()
