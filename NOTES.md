### Create a pixi workspace with the default environment from evironment.yaml
Initial python project
```
pixi init --format pyproject
```
Import conda evironment:
```
pixi import --format=conda-env environment.yaml --environment default # NAME MUST BE DEFAULT
```
Add entry point of python project
```
[project]
scripts = { cfclone = "cfclone.cli:main" }
```
run 
```
pixi run cfclone
```

### Add easy way to test via feature 
add task
```
pixi add --feature test pytest
```
We can then move this feature to its own environment
```
pixi workspace environment add test --feature test
```
```
pixi run --environment test pytest --version
```
to see the version of pytest installed.
Add to as task
```
pixi task add test --feature test test "pytest"
```
```
pixi run test
```

## Create workspace without pyproject format
```
pixi init --import environment.yaml
```