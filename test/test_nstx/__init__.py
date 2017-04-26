import socket
from fdp.classes import datasources as ds

# some valid shots for testing
shotlist = [204620, 204551, 141000, 204670, 204956, 204990]


def server_connection():
    machine = ds.machineAlias('nstx')
    servers = [ds.MDS_SERVERS[machine],
               ds.LOGBOOK_CREDENTIALS[machine]]
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
