# Dataset description
We use six public datasets for reproducibility. To be more specific,

Three NetFlow datasets:

1. [UGR16](https://nesg.ugr.es/nesg-ugr16/) dataset consists of traffic (including attacks) from NetFlow v9 collectors in a Spanish ISP network. We used data from the third week of March 2016. 
2. [CIDDS](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html) dataset emulates a small business environment with several clients and servers (e.g., email, web) with injected malicious traffic was executed. Each NetFlow entry recorded with the label (benign/attack) and attack type (DoS, brute force, port scan). 
3. [TON](https://research.unsw.edu.au/projects/toniot-datasets) dataset represents telemetry IoT sensors. We use a sub-dataset (“Train_Test_datasets”) for evaluating cybersecurity-related ML algorithms; of its 461,013 records, 300,000 (65.07%) are normal, and the rest (34.93%) combine nine evenly-distributed attack types (e.g., backdoor, DDoS, injection, MITM).

Three PCAP datasets:

1. [CAIDA](https://www.caida.org/catalog/datasets/passive_dataset/) contains anonymized traces from high-speed monitors on a commercial backbone link. Our subset is from the New York collector in March 2018. (**Require an CAIDA account to download the data**)
2. [DC](https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html) dataset is a packet capture from the "UNI1" data center studied in the [IMC 2010 paper](https://pages.cs.wisc.edu/~tbenson/papers/imc192.pdf).
3. [CA](https://www.netresec.com/?page=MACCDC) dataset is traces from The U.S. National CyberWatch Mid-Atlantic Collegiate Cyber Defense Competitions from March 2012.