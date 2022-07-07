#!/bin/bash

## Suppose that you are inside local folder /dns

#### CHANGE THESE VARIABLES WHEN USER/CLOUDLAB CHANGES ####
USERNAME=yyucheng

# NUMSLAVES = # of nodes - 1
NUMSLAVES=20


# HOSTNAME (general)
# HOSTNAME=31node-utah.cloudmigration-pg0.utah.cloudlab.us
# HOSTNAME=dg-emulab-utah.cloudmigration-pg0.utah.cloudlab.us
# HOSTNAME=dg-emulab.cloudmigration.emulab.net
# HOSTNAME=dg-wisc.cloudmigration-pg0.wisc.cloudlab.us
# HOSTNAME=test-wisc.cloudmigration-pg0.wisc.cloudlab.us
HOSTNAME=dg-wisc.cloudmigration-pg0.wisc.cloudlab.us
# HOSTNAME=dg-clemson.cloudmigration-pg0.clemson.cloudlab.us
# HOSTNAME=dg-utah.cloudmigration-pg0.utah.cloudlab.us

###########################################################

SERVER=1

(while [  $SERVER -le $NUMSLAVES ]; do
	echo $SERVER;
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eno1 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth0 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth1 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth2 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth3 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp24s0f0 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp24s0f1 | grep 'inet 130.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	 
	 
	
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eno1 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eno2 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eth0 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eth1 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eth2 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig eth3 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig enp130s0f0 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node-$SERVER.$HOSTNAME "ifconfig enp130s0f1 | grep 'inet addr:10.10' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";



	ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eno1 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eno49 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth0 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth1 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth2 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth3 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp1s0f0 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp24s0f0 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp24s0f1 | grep 'inet 128.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";


	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eno1 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth0 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth1 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth2 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig eth3 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp10s3f0 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";
	# ssh -o StrictHostKeyChecking=no  $USERNAME@node$SERVER.$HOSTNAME "ifconfig enp6s7 | grep 'inet 155.' | tr -s ' ' |  cut -d' ' -f3 | cut -d':' -f2";



    let SERVER=SERVER+1;
done) > test_measurer_ip.txt

# write slave ips to write to slave_servers.ini
python3 write_measurer_ip.py $NUMSLAVES
