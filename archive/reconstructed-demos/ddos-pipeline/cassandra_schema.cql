-- Optional schema corresponding to the remembered storage layer.
CREATE KEYSPACE IF NOT EXISTS ddos_data
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
CREATE TABLE IF NOT EXISTS ddos_data.network_traffic (
 id uuid PRIMARY KEY, highest_layer text, transport_layer text,
 src_ip text, dest_ip text, src_port int, dest_port int,
 packet_length int, packets_per_time double, target int
);
