#!/bin/bash
rm -f pcap2csv.so && cc -fPIC -shared -o pcap2csv.so main.c -lm -lpcap