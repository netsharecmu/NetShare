#!/bin/bash

USER=yyucheng
NUMHOSTS=21
EXPERIMENTNAME=netshare-wisc
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
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n env RESIZEROOT=192 bash -s" < grow-rootfs.sh
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "bash -s" < setup-cpu.sh "NetShare" & 
scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ~/.ssh/netshare-package $NODE_SYSTEM:~/.ssh/id_rsa &
pids+=($!)

# setup workers
COUNTER=1
while [  $COUNTER -lt $NUMHOSTS ]; do
    NODE="node${COUNTER}" 
    NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.${PROJECTNAME}.${LOCATION}.${SITE}"
    # NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.cloudmigration.emulab.net"
    echo $NODE_SYSTEM

    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n env RESIZEROOT=192 bash -s" < grow-rootfs.sh
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "bash -s" < setup-cpu.sh "NetShare" & 
    scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ~/.ssh/netshare-package $NODE_SYSTEM:~/.ssh/id_rsa &

    pids+=($!)
    let COUNTER=COUNTER+1
done

for pid in "${pids[@]}"; do
    wait "$pid"
done