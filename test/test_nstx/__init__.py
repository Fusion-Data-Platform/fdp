from __future__ import print_function
import socket
from fdp.lib import datasources

# some valid shots for testing
shotlist = [204620, 204551, 142301, 204670, 204956, 204990]


def server_connection():
    machine = datasources.canonicalMachineName('nstx')
    servers = [datasources.MDS_SERVERS[machine],
               datasources.LOGBOOK_CREDENTIALS[machine]]
    for server in servers:
        hostname = server['hostname']
        port = server['port']
        try:
            s = socket.create_connection((hostname, port), 3)
            s.close()
        except Exception as ex:
            print('Exception for host {} on port {}: {}'.format(hostname, port, ex))
            return False
    return True
