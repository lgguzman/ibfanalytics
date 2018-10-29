from subprocess import call
call("python3 -m pip install -r requirements.txt --upgrade --target dependencies", shell=True)
call("rm -r dependencies/*.dist-info", shell=True)
call("zip dependencies.zip dependencies", shell=True)


