# coding: utf-8

"""
Deployment file to facilitate releases.
"""

import os
import json
import webbrowser
import requests
from invoke import task
from amset import __version__
from monty.os import cd


@task
def make_doc(ctx):
    with cd("docs_rst"):
        ctx.run("sphinx-apidoc -o source/ -f ../amset")
        # ctx.run("make clean")
        ctx.run("make html")
        ctx.run("cp -r build/html/* ../docs")

    with cd("docs"):
        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")


@task
def update_doc(ctx):
    make_doc(ctx)
    with cd("docs"):
        ctx.run("git add .")
        ctx.run("git commit -a -m \"Update to v{}\"".format(__version__))
        ctx.run("git push")


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
        "prerelease": False
    }
    # For this to work properly, you need to go to your Github profile,
    # generate a "Personal access token". Then do export
    # GITHUB_RELEASES_TOKEN="xyz1234" (or add it to your bash_profile).
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/amset/releases",
        data=json.dumps(payload),
        headers={"Authorization":
                 "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def release(ctx, nosetest=False):
    if nosetest:
        ctx.run("nosetests")
    publish(ctx)
    update_doc(ctx)
    release_github(ctx)


@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)
