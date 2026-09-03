# Heartbeat.msg
# Broadcast periodically by every robot and the BS over the LoRa mesh.
# Used for neighbor discovery and RSSI tracking.

std_msgs/Header header    # timestamp + frame_id (which node sent this)

uint8 node_id             # unique ID of the sender (0 = BS, 1-255 = robots/victims)
float32 battery_pct       # battery percentage (0.0 to 100.0)
float64 latitude          # GPS latitude of sender
float64 longitude         # GPS longitude of sender
int16 rssi_dbm            # last measured RSSI in dBm (filled by receiver, not sender)