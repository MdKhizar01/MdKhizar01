"""Optional CSV-to-Cassandra loader for a local schema initialized separately."""
import argparse
import csv
import uuid
from cassandra.cluster import Cluster

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--host',default='127.0.0.1');a=p.parse_args()
    cluster=Cluster([a.host]);session=cluster.connect('ddos_data')
    insert=session.prepare('INSERT INTO network_traffic(id,highest_layer,transport_layer,src_ip,dest_ip,src_port,dest_port,packet_length,packets_per_time,target) VALUES(?,?,?,?,?,?,?,?,?,?)')
    try:
        count=0
        with open(a.csv,newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                session.execute(insert,[uuid.uuid4(),row.get('highest_layer'),row.get('transport_layer'),row.get('src_ip'),row.get('dest_ip'),int(row['src_port']),int(row['dest_port']),int(row['packet_length']),float(row['packets_per_time']),int(row['target'])])
                count+=1
        print(f'Inserted {count} rows')
    finally:cluster.shutdown()
