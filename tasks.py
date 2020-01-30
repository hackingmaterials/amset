# coding: utf-8

"""
Deployment file to facilitate releases.
"""

import json
import os

import requests
from invoke import task

from amset import __version__


@task
def publish(ctx):
    ctx.run("python setup.py release")


@task
def release_github(ctx):
    payload = {
        "tag_name": "v" + __version__,
        "target_commitish": "master",
        "name": "v" + __version__,
        "body": "",
        "draft": False,
        "prerelease": False,
    }
    # For this to work properly, you need to go to your Github profile,
    # generate a "Personal access token". Then do export
    # GITHUB_RELEASES_TOKEN="xyz1234" (or add it to your bash_profile).
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/amset/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)


@task
def release(ctx, nosetest=False):
    if nosetest:
        ctx.run("nosetests")
    publish(ctx)
    release_github(ctx)
