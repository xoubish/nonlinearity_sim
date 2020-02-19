#!/bin/bash

## List many files and put them to a file in JSON format

dir=$1
outfile=$2

#dir='/stage/euclid-staff-xliu/Nonlinearity/sim_data1'
#files=`ls -1v exposure_1utr*.fits` 
files=`ls -1v exposure1_utr{[0-9],[0-9][0-9],[0-9][0-9][0-9]}.fits`


echo "[" > $outfile

LAST_FILE=""
for f in $files
do
    if [ ! -z $LAST_FILE ]
    then
        #echo "Process file normally $LAST_FILE"
        echo \"${dir}/${LAST_FILE}\"\, >> $outfile 
    fi
    LAST_FILE=$f
done
if [ ! -z $LAST_FILE ]
then
    #echo "Process file as last file $LAST_FILE"
    echo \"${dir}/${LAST_FILE}\" >> $outfile
fi

echo "]" >> $outfile
 
