import socket 
from threading import Thread
import copy
import time
import json

class HuskyClient:
    # print_level = 0 off
    # print_level = 1 on
    # print_level = >1 on / print every nth mocap frame
    print_level = 20

    # Client/server message ids
    NAT_CONNECT               = 0
    NAT_SERVERINFO            = 1
    NAT_REQUEST               = 2
    NAT_RESPONSE              = 3
    NAT_REQUEST_MODELDEF      = 4
    NAT_MODELDEF              = 5
    NAT_REQUEST_FRAMEOFDATA   = 6
    NAT_FRAMEOFDATA           = 7
    NAT_MESSAGESTRING         = 8
    NAT_DISCONNECT            = 9
    NAT_KEEPALIVE             = 10
    NAT_UNRECOGNIZED_REQUEST  = 100
    NAT_UNDEFINED             = 999999.9999
    
    def __init__( self ):
        # Change this value to the IP address of the NatNet server.
        self.server_ip_address = "127.0.0.1"

        # Change this value to the IP address of your local network interface
        self.local_ip_address = "127.0.0.1"

        # This should match the multicast address listed in Motive's streaming settings.
        self.multicast_address = "239.255.42.99"

        # NatNet Command channel
        self.command_port = 1510

        # NatNet Data channel
        self.data_port = 1511

        self.use_multicast = False

        # Set this to a callback method of your choice to receive per-rigid-body data at each frame.
        self.joint_value_listener = None

        # Set Application Name
        self.__application_name = "Not Set"

        # Lock values once run is called
        self.__is_locked = False

        self.command_thread = None
        self.data_thread = None
        self.command_socket = None
        self.data_socket = None

        self.stop_threads=False


    def set_client_address(self, local_ip_address):
        if not self.__is_locked:
            self.local_ip_address = local_ip_address

    def get_client_address(self):
        return self.local_ip_address

    def set_server_address(self,server_ip_address):
        if not self.__is_locked:
            self.server_ip_address = server_ip_address

    def get_server_address(self):
        return self.server_ip_address

    def set_print_level(self, print_level=0):
        if(print_level >=0):
            self.print_level = print_level
        return self.print_level

    def get_print_level(self):
        return self.print_level


    def connected(self):
        ret_value = True
        # check sockets
        if self.command_socket == None:
            ret_value = False
        elif self.data_socket ==None:
            ret_value = False
        return ret_value


    # Create a command socket to attach to the NatNet stream
    def __create_command_socket( self ):
        result = None
        if self.use_multicast :
            # Multicast case
            result = socket.socket( socket.AF_INET, socket.SOCK_DGRAM, 0 )
            # allow multiple clients on same machine to use multicast group address/port
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                result.bind( ('', 0) )
            except socket.error as msg:
                print("ERROR: command socket error occurred:\n%s" %msg)
                print("Check Motive/Server mode requested mode agreement.  You requested Multicast ")
                result = None
            except  socket.herror:
                print("ERROR: command socket herror occurred")
                result = None
            except  socket.gaierror:
                print("ERROR: command socket gaierror occurred")
                result = None
            except  socket.timeout:
                print("ERROR: command socket timeout occurred. Server not responding")
                result = None
            # set to broadcast mode
            result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # set timeout to allow for keep alive messages
            result.settimeout(2.0)
        else:
            # Unicast case
            result = socket.socket( socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                result.bind( (self.local_ip_address, 0) )
            except socket.error as msg:
                print("ERROR: command socket error occurred:\n%s" %msg)
                print("Check Motive/Server mode requested mode agreement.  You requested Unicast ")
                result = None
            except socket.herror:
                print("ERROR: command socket herror occurred")
                result = None
            except socket.gaierror:
                print("ERROR: command socket gaierror occurred")
                result = None
            except socket.timeout:
                print("ERROR: command socket timeout occurred. Server not responding")
                result = None

            # set timeout to allow for keep alive messages
            result.settimeout(2.0)
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        return result

    # Create a data socket to attach to the NatNet stream
    def __create_data_socket( self, port ):
        result = None

        if self.use_multicast:
            # Multicast case
            result = socket.socket( socket.AF_INET,     # Internet
                                  socket.SOCK_DGRAM,
                                  0)    # UDP
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip_address))
            try:
                result.bind( (self.local_ip_address, port) )
            except socket.error as msg:
                print("ERROR: data socket error occurred:\n%s" %msg)
                print("  Check Motive/Server mode requested mode agreement.  You requested Multicast ")
                result = None
            except socket.herror:
                print("ERROR: data socket herror occurred")
                result = None
            except socket.gaierror:
                print("ERROR: data socket gaierror occurred")
                result = None
            except socket.timeout:
                print("ERROR: data socket timeout occurred. Server not responding")
                result = None
        else:
            # Unicast case
            result = socket.socket( socket.AF_INET,     # Internet
                                  socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #result.bind( (self.local_ip_address, port) )
            try:
                result.bind( ('', 0) )
            except socket.error as msg:
                print("ERROR: data socket error occurred:\n%s" %msg)
                print("Check Motive/Server mode requested mode agreement.  You requested Unicast ")
                result = None
            except socket.herror:
                print("ERROR: data socket herror occurred")
                result = None
            except socket.gaierror:
                print("ERROR: data socket gaierror occurred")
                result = None
            except socket.timeout:
                print("ERROR: data socket timeout occurred. Server not responding")
                result = None
            
            if(self.multicast_address != "255.255.255.255"):
                result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip_address))

        return result

    def __command_thread_function( self, in_socket, stop, gprint_level):
        message_id_dict={}
        if not self.use_multicast:
            in_socket.settimeout(2.0)
        data=bytearray(0)
        # 64k buffer size
        recv_buffer_size=64*1024
        while not stop():
            # Block for input
            try:
                data, addr = in_socket.recvfrom( recv_buffer_size )
            except socket.error as msg:
                if stop():
                    #print("ERROR: command socket access error occurred:\n  %s" %msg)
                    #return 1
                    print("shutting down")
            except  socket.herror:
                print("ERROR: command socket access herror occurred")
                return 2
            except  socket.gaierror:
                print("ERROR: command socket access gaierror occurred")
                return 3
            except  socket.timeout:
                if(self.use_multicast):
                    print("ERROR: command socket access timeout occurred. Server not responding")
                    #return 4

            if len( data ) > 0 :
                #peek ahead at message_id
                # if tmp_str not in message_id_dict:
                #     message_id_dict[tmp_str]=0
                # message_id_dict[tmp_str] += 1
                
                # print_level = gprint_level()
                # if message_id == self.NAT_FRAMEOFDATA:
                #     if print_level > 0:
                #         if (message_id_dict[tmp_str] % print_level) == 0:
                #             print_level = 1
                #         else:
                #             print_level = 0

                message_id = self.__process_message( data ) #, print_level)

                data=bytearray(0)

            if not self.use_multicast:
                if not stop():
                    self.send_keep_alive(in_socket, self.server_ip_address, self.command_port)
        return 0

    def __data_thread_function( self, in_socket, stop, gprint_level):
        message_id_dict={}
        data=bytearray(0)
        # 64k buffer size
        recv_buffer_size=64*1024

        while not stop():
            # Block for input
            try:
                data, addr = in_socket.recvfrom( recv_buffer_size )
            except socket.error as msg:
                if not stop():
                    print("ERROR: data socket access error occurred:\n  %s" %msg)
                    return 1
            except  socket.herror:
                print("ERROR: data socket access herror occurred")
                #return 2
            except  socket.gaierror:
                print("ERROR: data socket access gaierror occurred")
                #return 3
            except  socket.timeout:
                #if self.use_multicast:
                print("ERROR: data socket access timeout occurred. Server not responding")
                #return 4
            if len( data ) > 0 :
                #peek ahead at message_id
                # message_id = get_message_id(data)
                # tmp_str="mi_%1.1d"%message_id
                # if tmp_str not in message_id_dict:
                #     message_id_dict[tmp_str]=0
                # message_id_dict[tmp_str] += 1
                
                # print_level = gprint_level()
                # if message_id == self.NAT_FRAMEOFDATA:
                #     if print_level > 0:
                #         if (message_id_dict[tmp_str] % print_level) == 0:
                #             print_level = 1
                #         else:
                #             print_level = 0
                message_id = self.__process_message( data ) #, print_level)

                data=bytearray(0)
        return 0

    def __unpack_joint_value_data( self, data):
        data = memoryview( data )
        offset = 0
        data_json = json.loads(data[offset:].decode("utf-8"))

        if self.joint_value_listener is not None:
            self.joint_value_listener(data)

    def __process_message( self, data : bytes, print_level=0):
        # show_nat_net_version = False
        # if show_nat_net_version:
        #     trace("NatNetVersion " , str(self.__nat_net_requested_version[0]), " "\
        #         , str(self.__nat_net_requested_version[1]), " "\
        #         , str(self.__nat_net_requested_version[2]), " "\
        #         , str(self.__nat_net_requested_version[3]))
        # message_id = get_message_id(data)
        message_id = 0
        # packet_size = int.from_bytes( data[2:4], byteorder='little' )
        # print( "Message ID  : %3.1d NAT_FRAMEOFDATA"% message_id )
        # print( "Packet Size : ", packet_size )

        #skip the 4 bytes for message ID and packet_size
        offset = 0
        self.__unpack_joint_value_data( data[offset:])

        return message_id

    def send_request( self, in_socket, command, command_str, address ):
        # Compose the message in our known message format
        packet_size = 0
        if command == self.NAT_REQUEST_MODELDEF or command == self.NAT_REQUEST_FRAMEOFDATA :
            packet_size = 0
            command_str = ""
        elif command == self.NAT_REQUEST :
            packet_size = len( command_str ) + 1
        elif command == self.NAT_CONNECT :
            command_str = "Ping"
            packet_size = len( command_str ) + 1
        elif command == self.NAT_KEEPALIVE:
            packet_size = 0
            command_str = ""

        data = command.to_bytes( 2, byteorder='little' )
        data += packet_size.to_bytes( 2, byteorder='little' )

        data += command_str.encode( 'utf-8' )
        data += b'\0'

        return in_socket.sendto( data, address )

    def send_command( self, command_str):
        nTries = 3
        ret_val = -1
        while nTries:
            nTries -= 1
            ret_val = self.send_request( self.command_socket, self.NAT_REQUEST, command_str,  (self.server_ip_address, self.command_port) )
            if (ret_val != -1):
                break;
        return ret_val

        #return self.send_request(self.data_socket,    self.NAT_REQUEST, command_str,  (self.server_ip_address, self.command_port) )

    def send_commands(self, tmpCommands, print_results: bool =True):
        for sz_command in tmpCommands:
            return_code = self.send_command(sz_command)
            if(print_results):
                print("Command: %s - return_code: %d"% (sz_command, return_code) )

    def send_keep_alive(self,in_socket, server_ip_address, server_port):
        return self.send_request(in_socket, self.NAT_KEEPALIVE, "", (server_ip_address, server_port))


    def run( self ):
        # Create the data socket
        self.data_socket = self.__create_data_socket( self.data_port )
        if self.data_socket is None :
            print( "Could not open data channel" )
            return False

        # Create the command socket
        self.command_socket = self.__create_command_socket()
        if self.command_socket is None :
            print( "Could not open command channel" )
            return False
        self.__is_locked = True

        self.stop_threads = False
        # Create a separate thread for receiving data packets
        self.data_thread = Thread( target = self.__data_thread_function, args = (self.data_socket, lambda : self.stop_threads, lambda : self.print_level, ))
        self.data_thread.start()

        # Create a separate thread for receiving command packets
        self.command_thread = Thread( target = self.__command_thread_function, args = (self.command_socket, lambda : self.stop_threads, lambda : self.print_level,))
        self.command_thread.start()

        # Required for setup
        # Get NatNet and server versions
        self.send_request(self.command_socket, self.NAT_CONNECT, "",  (self.server_ip_address, self.command_port) )


        ##Example Commands
        ## Get NatNet and server versions
        #self.send_request(self.command_socket, self.NAT_CONNECT, "", (self.server_ip_address, self.command_port) )
        ## Request the model definitions
        #self.send_request(self.command_socket, self.NAT_REQUEST_MODELDEF, "",  (self.server_ip_address, self.command_port) )
        return True

    def shutdown(self):
        print("shutdown called")
        self.stop_threads = True
        # closing sockets causes blocking recvfrom to throw
        # an exception and break the loop
        self.command_socket.close()
        self.data_socket.close()
        # attempt to join the threads back.
        self.command_thread.join()
        self.data_thread.join()

