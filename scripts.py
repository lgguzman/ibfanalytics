from subprocess import call
call("python3 -m pip install -r requirements.txt --upgrade --target src/libs", shell=True)
call("mkdir ./dist", shell=True)
call("cp ./src/main.py ./dist", shell=True)
call("cd ./src && zip -x main.py -x \*libs\* -r ../dist/ibfanalytics.zip .", shell=True)
call("cd ./src/libs && zip -r ../../dist/libs.zip .", shell=True)


