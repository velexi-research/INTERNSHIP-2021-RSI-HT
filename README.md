Velexi Template: Data Science Project (v0.2.0)
==============================================

___Authors___  
Kevin T. Chu `<kevin@velexi.com>`

------------------------------------------------------------------------------

Table of Contents
-----------------

1. [Overview][#1]

   1.1. [Software Dependencies][#1.1]

   1.2. [Directory Structure][#1.2]

2. [Usage][#2]

   2.1. [Setting Up][#2.1]

   2.2. [Conventions][#2.2]

   2.3. [Environment][#2.3]

   2.4. [Using JupyterLab][#2.4]

3. [References][#3]

------------------------------------------------------------------------------

## 1. Overview

This project template is intended to support data science projects that
utilize Jupyter notebooks for experimentation and reporting. The design of
the template is based on the blog article
["Jupyter Notebook Best Practices for Data Science"][#whitmore-2016] by
Jonathan Whitmore.

Features include:

* compatible with standard version control software;

* automatically saves HTML and `*.py` versions of Jupyter notebooks to
  facilitate review of both (1) data science results and (2) implementation
  code;

* supports common data science workflows (for both individuals and teams); and

* encourages separation and decoupling of datasets, R&D work (i.e.,
  Jupyter notebooks), deliverables (i.e., reports), and Python functions and
  modules refactored from R&D code.

### 1.1. Software Dependencies

#### Base Requirements

* Python (>=3.7)

#### Optional Packages

* Miniconda
  * Required for MLflow Projects and MLflow Models
* Julia (>=1.6)
* `direnv`

### 1.2. Directory Structure

    README.md
    LICENSE
    README.md.template
    RELEASE-NOTES.md.template
    LICENSE.template
    requirements.txt
    Project.toml
    Manifest.toml
    bin/
    data/
    lib/
    reports/
    research/
    template-docs/
    template-docs/extras/

* `README.md`: this file (same as `README-Data-Science-Project-Template.md` in
  the `template-docs` directory)

* `LICENSE`: license for Data Science Project Template (same as
  `LICENSE-Data-Science-Project-Template.md` in the `template-docs` directory)

* `*.template`: template files for the package

    * Template files are indicated by the `template` suffix and contain
      template parameters denoted by double braces (e.g. `{{ PKG_NAME }}`).
      Template files are intended to simplify the set up of the package. When
      used, they should be renamed to remove the `template` suffix.

* `requirements.txt`: `pip` requirements file containing Python packages for
  project (e.g., data science, testing, and assessing code quality packages)

* `Project.toml`: Julia package management file containing Julia package
  dependencies. It is updated whenever new Julia packages are added via the
  REPL. This file may be safely removed if Julia is not required for the
  project.

* `Manifest.toml` (generated by Julia): Julia package management file that
  Julia uses to maintain a record of the state of the Julia environment. This
  file should _not_ be edited.

* `bin`: directory where scripts and programs should be placed

* `data`: directory where project data should be placed

    * __Recommendation__: data placed in the `data` directory should be managed
      using DVC (or a similar tool) rather than being included in the `git`
      repository. This is especially important for projects with large datasets
      or datasets containing sensitive information. For projects with small
      datasets that do not contain sensitive information, it may be reasonable
      to have the data contained in the `data` directory be managed directly by
      `git`.

* `lib`: directory containing source code to support the project (e.g.,
  custom code developed for the project, utility modules, etc.)

* `reports`: directory containing reports (in any format) that summarize
  research results. When a report is prepared as a Jupyter notebook, the
  notebook should be polished, contain final analysis results (not preliminary
  results), and is usually the work product of the entire data science team.

* `research`: directory containing Jupyter notebooks used for research phase
  work (e.g., exploration and development of ideas, DS/ML experiments). Each
  Jupyter notebook in this directory should (1) be dated and (2) have the
  initials of the person who last modified it. When an existing notebook is
  modified, it should be saved to a new file with a name based on the
  modification date and initialed by the person who modified the notebook.

* `template-docs`: directory containing documentation this package template

    * `template-docs/extras`: directory containing example and template files

------------------------------------------------------------------------------

## 2. Usage

### 2.1. Setting Up

1. Set up environment for project using only one of the following approaches.

  * `direnv`-based setup

    * Copy `template-docs/extras/envrc.template` to `.envrc` in project root
      directory.

    * Grant permission to `direnv` to execute the `.envrc` file.

      ```shell
      $ direnv allow
      ```

    * If needed, edit "User-Specified Configuration Parameters" section of
      `.envrc`.

  * `autoenv`-based setup

    * Create Python virtual environment.

      ```shell
      $ python3 -m venv .venv
      ```

    * Copy `template-docs/extras/env.template` to `.env` in project root
      directory.

    * If needed, edit "User-Specified Configuration Parameters" section of
      `.env`.

2. Install required Python packages.

    * If using cloud-based storage for DVC, modify the `dvc` line in
      `requirements.txt` to include the extra packages required to support
      the cloud-based storage.

      ```
      # DVC with S3 for remote storage
      dvc[s3]

      # DVC with Azure for remote storage
      dvc[azure]
      ```

    * Use `pip` to install Python packages.

      ```shell
      $ pip install -r requirements.txt
      ```

* (OPTIONAL) Set up Julia environment.

      ```shell
      $ julia

      julia> ]

      (...) pkg> instantiate
      ```

* (OPTIONAL) Set up DVC.

    * Initialize DVC.

      ```shell
      $ dvc init
      ```

    * Stop tracking `data` directory with `git`.

      ```shell
      $ git rm -r --cached 'data'
      $ git commit -m "Stop tracking 'data' directory"
      $ rm data/.git-keep-dir
      ```

3. Rename all of the template files with the `template` suffix removed
   (overwrite the original `README.md` and `LICENSE` files) and replace all
   template parameters with package-appropriate values.

4. Clean up project.

    * If Julia is not required for the project, remove `Project.toml` from the
      project.

### 2.2. Conventions

#### `research` directory

* Jupyter notebooks in the `research` directory should be named using the
  following convention: `YYYY-MM-DD-AUTHOR_INITIALS-BRIEF_DESCRIPTION.ipynb`.

  * Example: `2019-01-17-KC-information_theory_analysis.ipynb`

* Depending on the nature of the project, it may be useful to organize
  notebooks into sub-directories (e.g., by team member, by sub-project).

### 2.3. Environment

If `direnv` or `autoenv` is enabled, the following environment variables are
automatically set.

* `DATA_DIR`: absolute path to `data` directory

### 2.4. Using JupyterLab

* Launching a JupyterLab.

    ```shell
    $ jupyter-lab
    ```

* Use the GUI to create Jupyter notebooks, edit and run Jupyter notebooks,
  manage files in the file system, etc.

------------------------------------------------------------------------------

## 3. References

* J. Whitmore.
  ["Jupyter Notebook Best Practices for Data Science"][#whitmore-2016]
  (2016/09).

------------------------------------------------------------------------------

[-----------------------------INTERNAL LINKS-----------------------------]: #

[#1]: #1-overview
[#1.1]: #11-software-dependencies
[#1.2]: #12-directory-structure

[#2]: #2-usage
[#2.1]: #21-setting-up
[#2.2]: #22-conventions
[#2.3]: #23-environment
[#2.4]: #24-using-jupyterlab

[#3]: #3-references

[-----------------------------EXTERNAL LINKS-----------------------------]: #

[#whitmore-2016]:
  https://www.svds.com/tbt-jupyter-notebook-best-practices-data-science/
