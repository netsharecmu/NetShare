import sys

# argv[1]: NUMSLAVES
def main(argv):
	numSlaves = int(argv[1])

	fin = open("test_measurer_ip.txt", 'r')
	lines = fin.readlines()
	fin.close()

	dict_slave_ip = {}
	slave_No = -1
	slave_ip = ""
	for i, line in enumerate(lines):
		line = line.strip()
		if i%2 == 0:
			slave_No = int(line)
		else:
			slave_ip = str(line)
			dict_slave_ip[slave_No] = slave_ip


	fout = open("measurers.ini", 'w')
	fout.write("[measurer]\n")
	fout.write("numMeasurers=%d\n" %numSlaves)
	fout.write("\n")

	for key, value in dict_slave_ip.items():
		fout.write("[Measurer_%d]\n" %key)
		fout.write("ip=%s\n" %value)



	fout.close()

if __name__ == "__main__":
	main(sys.argv)