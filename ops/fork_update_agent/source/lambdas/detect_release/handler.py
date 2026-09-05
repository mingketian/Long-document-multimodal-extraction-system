import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import boto3


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

SSM = boto3.client("ssm")


def _fetch_github(path: str, token: str | None = None) -> dict[str, Any]:
    url = f"https://api.github.com{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 (GitHub endpoint)
        body = response.read()
        return json.loads(body)


def _latest_release(owner: str, repo: str, token: str | None = None) -> dict[str, Any]:
    try:
        payload = _fetch_github(f"/repos/{owner}/{repo}/releases/latest", token)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            LOGGER.warning("Release API returned 404. Falling back to tags endpoint.")
            tags = _fetch_github(f"/repos/{owner}/{repo}/tags", token)
            if not tags:
                raise RuntimeError("No tags returned from GitHub.") from err
            payload = {
                "tag_name": tags[0]["name"],
                "html_url": f"https://github.com/{owner}/{repo}/tree/{tags[0]['name']}",
                "body": "",
                "published_at": tags[0].get("commit", {}).get("date"),
            }
        else:
            raise
    return payload


def _get_parameter(name: str, decrypt: bool = False) -> str:
    response = SSM.get_parameter(Name=name, WithDecryption=decrypt)
    return response["Parameter"]["Value"]


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    LOGGER.info("Event received: %s", event)
    owner = os.environ["UPSTREAM_OWNER"]
    repo = os.environ["UPSTREAM_REPO"]
    fork_repo = os.environ["FORK_REPO"]
    token_param = os.environ["GITHUB_TOKEN_PARAM"]
    version_param = os.environ["CURRENT_VERSION_PARAM"]

    token = None
    if token_param:
        try:
            raw_token = _get_parameter(token_param, decrypt=True)
            if raw_token and raw_token != "REPLACE_ME":
                token = raw_token
            else:
                LOGGER.info("GitHub token is placeholder. Continuing without auth.")
        except SSM.exceptions.ParameterNotFound:
            LOGGER.warning("GitHub token parameter %s not found. Continuing without auth.", token_param)

    release = _latest_release(owner, repo, token)
    latest_version = release.get("tag_name") or release.get("name")

    if not latest_version:
        raise RuntimeError("Unable to determine latest version from GitHub payload.")

    try:
        current_version = _get_parameter(version_param)
    except SSM.exceptions.ParameterNotFound:
        current_version = "0.0.0"

    update_required = current_version != latest_version

    LOGGER.info(
        "Current version: %s | latest: %s | update_required=%s", current_version, latest_version, update_required
    )

    return {
        "update_required": update_required,
        "current_version": current_version,
        "upstream_version": latest_version,
        "release_url": release.get("html_url"),
        "published_at": release.get("published_at"),
        "release_notes": release.get("body", ""),
        "fork_repo": fork_repo,
        "owner": owner,
        "repo": repo,
    }
