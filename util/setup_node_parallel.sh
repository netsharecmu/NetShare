#!/bin/bash

USER=minhao
NUMHOSTS=16
EXPERIMENTNAME=netshare-ray-emu
PROJECTNAME=cloudmigration
# LOCATION=utah
LOCATION=emulab
# LOCATION=clemson
SITE=net

pids=()

# setup controller
NODE_SYSTEM="${USER}@nfs.${EXPERIMENTNAME}.${PROJECTNAME}.${LOCATION}.${SITE}"
# NODE_SYSTEM="${USER}@nfs.${EXPERIMENTNAME}.cloudmigration.emulab.net"
echo $NODE_SYSTEM
# ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n env RESIZEROOT=192 bash -s" < grow-rootfs.sh
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "bash -s" < setup-cpu.sh "NetShare" $USER &
scp -r -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ~/.ssh/netshare-ray-emu/id_rsa $NODE_SYSTEM:~/.ssh/ &
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "chmod 600 ~/.ssh/id_rsa" & 
pids+=($!)

setup workers
COUNTER=1
while [  $COUNTER -lt $NUMHOSTS ]; do
    NODE="node${COUNTER}" 
    NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.${PROJECTNAME}.${LOCATION}.${SITE}"
    # NODE_SYSTEM="${USER}@${NODE}.${EXPERIMENTNAME}.cloudmigration.emulab.net"
    echo $NODE_SYSTEM

    # ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "sudo -n env RESIZEROOT=192 bash -s" < grow-rootfs.sh
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "bash -s" < setup-cpu.sh "NetShare" $USER &
    scp -r -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ~/.ssh/netshare-ray-emu/id_rsa $NODE_SYSTEM:~/.ssh/ &
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $NODE_SYSTEM "chmod 600 ~/.ssh/id_rsa" & 

    pids+=($!)
    let COUNTER=COUNTER+1
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
