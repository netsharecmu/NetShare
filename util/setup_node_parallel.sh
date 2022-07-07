#!/bin/bash

USER=yyucheng
NUMHOSTS=21
# EXPERIMENTNAME=dg-utah
EXPERIMENTNAME=dg-wisc
# EXPERIMENTNAME=dg-emulab
# EXPERIMENTNAME=dg-clemson
PROJECTNAME=cloudmigration-pg0
# LOCATION=utah
LOCATION=wisc
# LOCATION=clemson
SITE=cloudlab.us

pids=()

# setup controller
NODE_SYSTEM="${USER}@nfs.${EXPERIMENTNAME}.${PROJECTNAME}.${LOCATION}.${SITE}"
# NODE_SYSTEM="${USER}@nfs.${EXPERIMENTNAME}.cloudmigration.emulab.net"
echo $NODE_SYSTEM
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n bash -s" < setup-cpu.sh & 
pids+=($!)

# setup workers
COUNTER=1
while [  $COUNTER -lt $NUMHOSTS ]; do
    NODE="node${COUNTER}" 
    NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.${PROJECTNAME}.${LOCATION}.${SITE}"
    # NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.cloudmigration.emulab.net"
    echo $NODE_SYSTEM

    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n bash -s" < setup-cpu.sh & 
    pids+=($!)
    let COUNTER=COUNTER+1
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
