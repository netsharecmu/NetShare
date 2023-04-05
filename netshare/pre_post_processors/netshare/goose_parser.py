import csv
from scapy.all import *
from scapy.layers.l2 import Ether, Dot1Q


class GOOSE(Packet):
    name = "GOOSE"
    fields_desc = [
        XShortField("appid", 0x0000),
        XShortField("length", None),
        XIntField("resv", 0x00000001),
        XByteField("gocbRef_len", None),
        StrLenField("gocbRef", "", length_from=lambda pkt: pkt.gocbRef_len),
        XIntField("timeAllowedtoLive", 0x00000000),
        XIntField("datSet_len", None),
        StrLenField("datSet", "", length_from=lambda pkt: pkt.datSet_len),
        XIntField("goID_len", None),
        StrLenField("goID", "", length_from=lambda pkt: pkt.goID_len),
        XLongField("t", 0x0000000000000000),
        XShortField("stNum", 0x0000),
        XShortField("sqNum", 0x0000),
        XIntField("test", 0x00000000),
        XShortField("confRev", 0x0000),
        XShortField("ndsCom", 0x0000),
        XIntField("numDatSetEntries", 0x00000000),
        # Additional fields can be added for the specific structure of your GOOSE payload
    ]


# Bind the GOOSE layer to Ether and Dot1Q layers
bind_layers(Ether, GOOSE, type=0x8100)
bind_layers(Dot1Q, GOOSE, type=0x88b8)


def process_packet(packet, csv_writer):
    if packet.haslayer(GOOSE):
        print("GOOSE packet found!")
        packet.show()

        timestamp = packet.time
        src_mac = packet[Ether].src
        dst_mac = packet[Ether].dst
        length = packet[GOOSE].length

        csv_writer.writerow([timestamp, src_mac, dst_mac, length])


def main():
    pcap_file = '../../IEC61850SecurityDataset/Normal/No_Variable_Loading/Normal.pcapng'
    output_csv_file = '../../IEC61850SecurityDataset/Normal/No_Variable_Loading/Normal.csv'
    print(
        f"Parsing GOOSE packets from {pcap_file} and exporting data to {output_csv_file}")

    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Timestamp', 'SourceMAC', 'DestinationMAC', 'Length'])

        sniff(
            offline=pcap_file,
            prn=lambda packet: process_packet(packet, csv_writer),
            store=0)
