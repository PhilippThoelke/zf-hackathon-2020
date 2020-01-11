#!/bin/bash
while read P1 P2 P3
do
export C="$P1"
export vel="$P2"
export roadProfile="$P3"

qsub -v C -v vel -v roadProfile runDamper.sge
done < params.txt
exit
