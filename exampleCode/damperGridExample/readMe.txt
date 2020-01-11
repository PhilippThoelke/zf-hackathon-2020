this grid example can be run by ./submit_runDamper.sh

before executing
1) copy folder to your folder
2) change the location of your python environment in runDamper.sge (source source ~/hack20/bin/activate)
2) set the parameter in params.txt
3) make submit_runDamper.sh executable chmod +x
4) run ./submit_runDamper.sh

Hint: if you doing adjustment your should uncomment "#$ -o /dev/null" and ""#$ -e /dev/null" in order to get printouts
