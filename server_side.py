import socket
import argparse
import struct
import binascii

BUFFER_SIZE = 65536

def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

class DynamicImporter:
    """"""
 
    #----------------------------------------------------------------------
    def __init__(self, module_name, class_name):
        """Constructor"""
        module = __import__(module_name)
        my_class = getattr(module, class_name)
        instance = my_class()
        return instance

class PowerSystem:
	def __init__(self):
		self.rtds_data=[]
		self.case_dict={
			0:'case4gs',
			1:'case6ww',
			2:'case9',
			3:'case9q',
			4:'case14',
			5:'case24',
			6:'case30',
			7:'case30q',
			8:'case30pwl',
			9:'case39',
			10:'case57',
			11:'case118',
			12:'case300'
		}
		self.case=0
		self.port=12001
		self.ppc={}
		self.measurements={}
		self.observerability_matrix=[]
		self.overall_availability_vector=[]

	def process_input(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--port',dest=port,help='Port number to listen on')
		parser.add_argument('--c1',dest=cb_na_data_av_flag,help='If Circuit Breaker status is unavailable, and corresponding Power flow data is available, then c1 = 0 discards PF data, and any other value of c1 assumes CB is ON, and uses data',type=int)
		parser.add_argument('--c2',dest=cb_off_data_av_flag,help='If Circuit Breaker status is OFF (Line Disconnected), and corresponding Power flow data is available, then c2 = 0 discards PF data, and any other value of c2 assumes CB is ON (Line Connected), and uses data',type=int)
		parser.add_argument('--c3',dest=cb_off_data_na_flag,help='If Circuit Breaker status is ON (Line Connected), and corresponding Power flow data is unavailable, then c3 = 0 discards PF data, and any other value of c2 assumes CB is ON (Line Connected), and uses data',type=int)
		args=parser.parse_args()
		self.port=args.port

	def get_data_from_rtds(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind(('', self.port))
		self.rtds_data=[]
		while True:
			data, addr = s.recvfrom(BUFFER_SIZE)
			if len(data) != 0:
				values=data[i:i+4] for i in range(0,len(data),4)
				for value_ascii_str in values:
					value_hex_str = bin(int(binascii.hexillify(value_ascii_str),16))
					f=int(value_hex_str,2)
					self.rtds_data.add(struct.unpack('f',struct.pack('I',f))[0])
				break

	def extract_ppc_from_rtds_data(self):
		assert self.rtds_data is not None and len(self.rtds_data) != 0
		if not self.rtds_data[0].is_integer():
			print("The first input from RTDS needs to be a whole number, in floating point format from RTDS, denoting the case number for the system")
			exit(0)
		# Note: All the data that RTDS sends needs to be in float format, for proper processing by this code
		#first number of rtds_data is the system case (float),
		# all these are the formats given by PyPower, based on MATPOWER. PyPower (https://github.com/rwl/PYPOWER/tree/master/pypower)


		file_to_read=case_dict[int(self.rtds_data[0])]
		self.ppc=DynamicImporter(file_to_read,file_to_read)

	# def extract_circuit_breaker_status(self):
	# 	''' from first line we know the number of circuit breakers, so if that number is > 32, we need GIF(number of circuit breakers/32) + 1 numbers to represent the availability of measurements
	# 	We will read msb to lsb for bits, and for the last number, ignore the remaining bits.
	# 	'''

	# 	# Ensure that RTDS data has been extracted and that a valid topology has been identified
	# 	assert (self.rtds_data is not None and len(self.rtds_data) != 0) and (len(self.ppc) != 0)


	# 	branch_data=self.ppc["branch"]
	# 	self.max_num_cb=2*len(branch_data)
	# 	'''
	# 	For n circuit breakers, you need n bits to encode their status.
	# 	Each value sent by RTDS is a 32 bit floating point number, therefore it can encode status of maximum 32 circuit breakers
	# 	For n > 32, you need more than 1 floating point number to encode the status, and the following variable encodes that quantity.
	# 	The data is sent as follows: Let n=40, then the first floating point number will have 32 bits for the status of the first 32 CB.
	# 	The next floating point number will have first 8 bits telling the status of the remaining 8 CB, and the rest of the bits will be 0
	# 	'''

	# 	self.num_cb_decimals_req=int(max_num_ckt_breakers/32) if (max_num_ckt_breakers%32 == 0) else ((max_num_ckt_breakers/32)+1)


	# 	availability_ckt_brkrs=[]

	# 	for i in range(num_ckt_brk_status_nums_req):
	# 		bin_str=binary(rtds_data[1+i])
	# 		for b in bin_str:
	# 			availability_ckt_brkrs.add(int(b)) # 1 if available, 0 if not

	# 	availability_ckt_brkrs=availability_ckt_brkrs[:max_num_ckt_breakers] # only the <max_number_of_circuit_breakers> bits are useful, rest is just padding


	# 	num_available_ckt_brkrs=sum(status_ckt_brkrs)

	# 	assert num_available_ckt_brkrs <=max_num_ckt_breakers

	# 	#position update, used in rtds_data
	# 	current_pos=1+num_ckt_brk_status_nums_req
	# 	ckt_brk_pos=0
	# 	'''Modify availability_vector of circuit breakers to the following
	# 	-1 if unavailable, 0 if line is closed, 1 if line is open
	# 	'''
	# 	availability_ckt_brkrs=[i-1 for i in availability_ckt_brkrs] 


	# 	for i in range(num_available_ckt_brkrs):
	# 		cur_val=rtds_data[current_pos+i]
	# 		while availability_ckt_brkrs[ckt_brk_pos] != 0:
	# 			ckt_brk_pos+=1

	# 		if availability_ckt_brkrs[ckt_brk_pos] == 0:
	# 			availability_ckt_brkrs[ckt_brk_pos]=cur_val
	# 			ckt_brk_pos+=1
	# 	current_pos+=num_available_ckt_brkrs

	# 	self.cb_status=availability_ckt_brkrs
	# 	return current_pos # useful when parsing for Pbus, Qbus, Pflow and Qflow

	def extract_measurement_data(self,start_pos=1,measurement_case=0):
		'''General function to extract data from rtds_data
		Parameters: start_pos: the starting index in the rtds_data list, from where to begin reading the status decimals. Required because measurements come one after other, so need to keep track of data
		measurement_case:
		0 = circuit breakers
		1 = Pbus
		2 = Qbus
		3 = Pflow
		4 = Qflow
		Returns: a list: Values can be np.nan if unavailable, or the values read from rtds_data if available,
		and updated_pos: the value from which to continue reading the next set of measurements
		'''
		assert (self.rtds_data is not None and len(self.rtds_data) != 0) and (len(self.ppc) != 0)
		num_branches=len(self.ppc['branch'])
		num_buses=len(self.ppc['bus'])

		max_measurements = [ (2*num_branches) , num_buses , num_buses, num_branches, num_branches ] # for CB, Pbus, Qbus, Pflow, Qflow
		num_decimals_req = [int(i/32) if (i%32 == 0) else ((i/32)+1) for i in max_measurements] # number of 32 bit strings needed to capture status of measurements

		'''parse data, starting from start_pos
		'''
		availability_vector=[]
		for i in range(num_decimals_req[measurement_case]):
			bin_str=binary(rtds_data[start_pos+i])
			for b in bin_str:
				availability_vector.add(int(b)) # 1 if available, 0 if not

		availability_vector=availability_vector[:max_measurements[measurement_case]] # rest are padding (0 values)
		start_pos+=num_decimals_req[measurement_case]

		num_measurements_available=sum(availability_vector)

		assert num_measurements_available <= max_measurements[measurement_case]

		position=0

		availability_vector=[np.nan if i == 0 else 0 for i in availability_vector]
		# None if unavailable, 0 if available. 0 overwritten by measurement

		for i in range(num_measurements_available):
			cur_val=rtds_data[start_pos+i]

			while np.isnan(availability_vector[position]):
				position+=1

			if availability_vector[position] == 0:
				availability_vector[position]=cur_val
				position+=1

		start_pos+=num_measurements_available
		assert position == len(availability_vector)

		return availability_vector,start_pos

	def parse_rtds_data(self):
		'''Calls extract_measurement_data for each measurement type
		'''
		self.extract_ppc_from_rtds_data()
		start_pos=1
		self.measurements = {}
		keys_arr=['cb','pbus','qbus','pflow','qflow']
		case_readings=[]
		self.overall_availability_vector=[]
		for measurement_case in range(5):
			case_readings,start_pos = extract_measurement_data(measurement_case= measurement_case, start_pos = start_pos)
			
			self.measurements[keys_arr[measurement_case]]=case_readings
			
			if measurement_case >=1:
				self.overall_availability_vector.append([1 if not np.isnan(i) else 0 for i in case_readings])


	def create_observability_topology_matrix(self):
		baseMVA=self.ppc["baseMVA"]
		bus_data=self.ppc["bus"]
		gen_data=self.ppc["gen"]
		branch_data=self.ppc["branch"]
		gen_cost_data=self.ppc["gencost"]
		buses=[bus_data[i][0] for i in range(len(bus_data))]
		num_buses=len(buses)
		lines=[min([line[0]-1,line[1]-1])*num_buses+max([line[0]-1,line[1]-1]) for line in branch_data] # all lines are sorted in order <min(sending bus, receiving bus)>, then <max(sending bus, receiving bus)>. Case files also sort by this method
		self.observerability_matrix=np.zeros(shape=(2*num_buses+2*len(branch_data,num_buses),num_buses))
		'''
		First <num_buses> rows, for Pbus, then for Qbus
		Then <num_branches> for Pflow, Qflow
		'''
		for branches in branch_data:
			# For Pbus, update all neighbours of buses to 1
			# assuming all bus P,Q data are avaiable
			self.observerability_matrix[branches[0]-1][branches[1]-1]=1
			self.observerability_matrix[branches[1]-1][branches[0]-1]=1
			# For Qbus
			self.observerability_matrix[num_buses+branches[0]-1][branches[1]-1]=1
			self.observerability_matrix[num_buses+branches[1]-1][branches[0]-1]=1
		
		line_counter=2*len(buses)
		for l in lines:
			# For Pflow
			self.observerability_matrix[line_counter][l%num_buses]=1
			self.observerability_matrix[line_counter][l/num_buses]=1
			# For Qflow
			self.observerability_matrix[len(branch_data)+line_counter][l%num_buses]=1
			self.observerability_matrix[len(branch_data)+line_counter][l/num_buses]=1
			line_counter+=1

	def check_observability(self):
		self.parse_rtds_data()
		self.create_observability_topology_matrix()

		assert np.shape(self.observerability_matrix)[1] == np.shape(overall_availability_vector)[0]
		observability= np.dot(self.observerability_matrix,self.overall_availability_vector)

		for measurement_status in observability:
			if measurement_status == 0:
				return -1
				# un observable

		# observable
		return 0







baseMVA=self.ppc["baseMVA"]
bus_data=self.ppc["bus"]
gen_data=self.ppc["gen"]
branch_data=self.ppc["branch"]
gen_cost_data=self.ppc["gencost"]

max_num_gen=len(gen_data)
max_num_load=len(bus_data)
num_gen_status_nums_req= max_num_gen/32 if max_num_gen%32 == 0 else (max_num_gen/32+1)
num_load_status_nums_req= max_num_load/32 if max_num_load%32 == 0 else (max_num_load/32+1)

########################## getting the circuit breaker values ###############################



current_pos=1+num_ckt_brk_status_nums_req+num_available_ckt_brkrs

########################## getting the P injection values ###############################
availability_gen=[]
for i in range(num_gen_status_nums_req):
	bin_str=binary(rtds_data[current_pos+i])
	for b in bin_str:
		availability_gen.add(int(b)) # 1 if available, 0 if not

availability_gen=availability_gen[:max_num_gen]

num_available_gen=sum(availability_gen)

assert num_available_gen <= max_num_gen

current_pos+=num_gen_status_nums_req
gen_pos=0
availability_gen=[i-1 for i in availability_gen]

for i in range(num_available_gen):
	cur_val=rtds_data[current_pos+i]
	while availability_gen[gen_pos] != 0:
		gen_pos+=1
	if availability_gen[gen_pos] == 0:
status_p_gen=[]


