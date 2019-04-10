from requests import get
import multiprocessing
import os
import socket
import subprocess
import netifaces as ni
from netaddr import IPNetwork, IPAddress

def primary_nic_info():
    # Works on docker containers
    if 'eth0' in ni.interfaces():
        return ni.ifaddresses('eth0')[ni.AF_INET][0]

    # Fallback if eth0 does not exist.
    # Not the primary way, because docker's overlay network is not the default gateway
    else:
        nic = ni.gateways()['default'][ni.AF_INET][1]
        return ni.ifaddresses(nic)[ni.AF_INET][0]


def local_private_ip():
    return primary_nic_info()['addr']


def local_public_ip():
    # Some servers like AWS have both public and private IP addresses
    try:
        return get('https://api.ipify.org').text
    except:
        return None


def local_submask():
    return primary_nic_info()['netmask']


def is_local_host(address):
    # For servers like AWS, public IP is required for communciation between
    # different instances
    return address == 'localhost' or address == '127.0.0.1' \
            or address == local_private_ip() or address == local_public_ip()


def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind(("127.0.0.1", port))
    except socket.error:
        return False

    s.close()
    return True

def get_network_devices(pool_size=255, client_ips=[]):
    def pinger(job_q, results_q):
        dev_null = open(os.devnull, 'w')
        while True:
            ip = job_q.get()
            if ip is None:
                break
            try:
                subprocess.check_call(['ping', '-c1', ip],
                                      stdout=dev_null)
                results_q.put(ip)
            except Exception:
                pass
    ip_list = list()
    print("List of the proposed client IPs: {}".format(client_ips))
    if len(client_ips) > 0:
        list_ips = list()
        for client_ip in client_ips:
            list_ips.append(IPAddress(client_ip))
            pool_size = len(list_ips)
    else:
        network = '{}/{}'.format(local_private_ip(), local_submask())
        list_ips = IPNetwork(network)
    #listIps = [IPAddress('10.0.0.6')]
    #pool_size=len(listIps)
    print(pool_size)
    # prepare the jobs queue
    jobs = multiprocessing.Queue()
    results = multiprocessing.Queue()
    pool = [multiprocessing.Process(target=pinger, args=(jobs, results)) for _ in range(pool_size)]
    for p in pool:
        p.start()
    for ip in list_ips:
    #for ip in IPNetwork(network):
        jobs.put(str(ip))
    for _ in pool:
        jobs.put(None)
    for p in pool:
        p.join()
    # collect the results
    while not results.empty():
        ip_list.append(results.get())
    print(ip_list)
    return ip_list

    
