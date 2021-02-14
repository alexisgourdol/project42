# A simple EDA app on Streamlit :
- Project: EDA app on Streamlit

- Description:
  - (1) EDA app on Streamlit
  - (2) to filter data -- initial focus will be on categorical data and
  - (3) later prepare a matching system on two datasets sharing common columns

- Data Source: to be uploaded by the user on the UI
- Links:

## Startup the project

### The initial setup.
Clone repo

```bash
mkdir ~/code/alexisgourdol
cd ~/code/alexisgourdol
git clone git@github.com:alexisgourdol/project42.git
```

### Create virtualenv and install the project

Using `pyenv`
https://github.com/pyenv/pyenv#homebrew-on-macos

If you're on Windows, consider using @kirankotari's pyenv-win fork.
(pyenv does not work on windows outside the Windows Subsystem for Linux)

```bash
pyenv virtualenv p42 # create a new virtualenv for our project
pyenv virtualenvs           # list all virtualenvs
pyenv activate p42   # enable our new virtualenv
pip install --upgrade pip   # install and upgrade pip
pip list                    # list all installed packages
```

### Install `requirements.txt` :

```bash
pip install -r https://raw.githubusercontent.com/alexisgourdol/project42/master/requirements.txt
pip list
```

### Contributions : process / best practices

If you want to contribute, please let me know. Then:

1. Make sure your git status is clean
`git status`

2. Get latest master

`git checkout master`

`git pull origin master`


3. 1 task = 1 branch

`git checkout -b my-task`

Work on the existing files, or create new ones

`git add .`

`git commit -m "This is an informative message about my-task" `

`git push origin my-task`

4. Create a pull request

Use the github website

Click on compare & pull request

5. Someone else Reviews and Approves the pull request

6. Remove unused branches locally

`git sweep`
