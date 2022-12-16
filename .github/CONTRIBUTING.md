# Contributing

**Thank you for your interest in making Openseize better!**

- [Contributing Etiquette](#contributing-etiquette)
- [Creating an Issue](#creating-an-issue)
- [Creating a Pull Request](#creating-a-pull-request)
    * [Requirements](#requiirements)
    * [Setup](#setup)
    * [Modifications](#modifications)
    * [Updating Documentation](#update-documentation)
    * [Review Process](#review-process)
- [Commit Message Guide](#coomit-message-guide)


## Contributing Etiquette

Please see our [Contributor Code of Conduct](
https://github.com/mscaudill/openseize/blob/master/CODE_OF_CONDUCT.md) for
information on our rules of conduct.

## Creating an Issue

We have created [issue templates](
https://github.com/mscaudill/openseize/issues) for:

- Asking questions
- Filing bug reports
- Making feature requests
- Improving Openseize's documentation

These templates have required fields that will encourage you to be as
specific as possible in defining the issue you want help with. We ask that
you use these templates whenever possible.

## Creating a Pull Request

> Note: We appreciate you taking the time to contribute! Before starting
> a pull request please discuss with us in the comments of an open issue.
> This helps use to identify what issues are being worked on to prevent
> duplicate effort.

### Requirements

1. PRs must have a reference url to an existing issue that describes why the
   issue or feature needs addressing.
2. PRs must pass code quality checks with `pylint` and pass static type
   checking with `mypy` where appropriate.
3. PRs must have unit test with `pytest` that cover the changed behavior.

### Setup

1. Open an issue to discuss the changes you would like to see in Openseize.
2. Fork Openseize's master branch and create a local branch for your change.
3. Sumbit your fantastic PR!

### Modifications

When modifying Openseize, please follow [Google's code style](
https://google.github.io/styleguide/pyguide.html) for documenting modules,
classes and functions. Additionally, Openseize is transitioning to using
type annotations. When adding new code or modifying existing code that
lacks annotations, we ask that you use annotations or insert them as needed.
All pull request will have isort, pylint and mypy automatically run during
a merge so please run each of these code quality checkers/formatters on the
files you change prior to submitting your pull request. You can create
a development environment in conda or venv by using our development
environments defined in the [develop.yml](
https://github.com/mscaudill/openseize/blob/master/develop.yml) (conda) or
[pyproject.toml](
https://github.com/mscaudill/openseize/blob/master/pyproject.toml) (pip)
files.

### Updating Documentation

After adding to or modifying Openseize's source code, we ask that you update
the mkdocs [documentation pages](
https://github.com/mscaudill/openseize/tree/master/docs) and submit those
changes with your pull request. This will allow us to more quickly get your
new features integrated with Openseize.

### Review Process

To expedite a review of your pull request, we ask that:
1. you reference the issue url your pull request is addressing.
2. you comment modifications so that we can quickly understand your
   changes/additions.
3. you provide tests that demonstrate that your modifications work.

## Commit Message Guide

While building Openseize, we discovered a need for greater clarity in our
commit messages. This led us to discover the [angular](
https://gist.github.com/brianclements/841ea7bffdb01346392c) commit message
guide. We are starting to follow this guide and would encourage you to
follow it too. We have found it very helpful in sorting through previous
commits.





