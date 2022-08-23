#!/bin/bash
cc -fPIC -shared -o pcap2csv.so main.c -lm -lpcap